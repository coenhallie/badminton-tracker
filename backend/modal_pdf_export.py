"""
Modal PDF Export for Badminton Tracker (Convex Integration)

This module provides GPU-accelerated PDF report generation that:
1. Downloads video and results from Convex storage
2. Generates heatmaps and extracts frames
3. Creates a professional PDF report
4. Returns the PDF bytes
"""

import os
import json
import tempfile
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import modal


# =============================================================================
# MODAL CONFIGURATION
# =============================================================================

# Container image with all dependencies for PDF generation
pdf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        # System libraries for OpenCV and graphics
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        # Fonts for PDF
        "fonts-dejavu-core",
    ])
    .pip_install([
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "reportlab>=4.0.0",
        "httpx>=0.25.0",
        "pillow>=10.0.0",  # For image processing in PDF
        "fastapi[standard]>=0.104.0",  # Required for Modal web endpoints
    ])
)

# Initialize Modal app
app = modal.App(
    "badminton-tracker-pdf-export",
    image=pdf_image
)


# =============================================================================
# COLORS AND CONSTANTS
# =============================================================================

# Theme colors (RGB tuples for OpenCV, will be converted for ReportLab)
THEME_GREEN = (34, 197, 94)  # #22c55e
THEME_BLACK = (26, 26, 26)
THEME_GRAY = (102, 102, 102)
THEME_LIGHT_GRAY = (242, 242, 242)

# Player colors for heatmaps
PLAYER_COLORS = [
    (230, 64, 64),   # Red
    (38, 140, 133),  # Cyan
    (38, 128, 153),  # Blue
    (89, 153, 115),  # Green
]


# =============================================================================
# HEATMAP GENERATION (Inline - no external dependencies)
# =============================================================================

def generate_heatmap_overlay(
    skeleton_data: List[Dict],
    video_width: int,
    video_height: int,
    player_id: Optional[int] = None,
    colormap: str = "turbo"
) -> Any:
    """
    Generate a heatmap from player position data.
    
    Args:
        skeleton_data: List of frame data with player positions
        video_width: Video frame width
        video_height: Video frame height
        player_id: Optional player ID to filter (None = all players)
        colormap: Colormap name (turbo, jet, hot, etc.)
    
    Returns:
        Heatmap as numpy array (RGBA)
    """
    import numpy as np
    import cv2
    
    # Create accumulator for heatmap
    heatmap = np.zeros((video_height, video_width), dtype=np.float32)
    
    # Gaussian kernel parameters
    kernel_size = 51
    sigma = 25.0
    
    # Create Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Accumulate positions
    for frame_data in skeleton_data:
        players = frame_data.get("players", [])
        for player in players:
            pid = player.get("player_id", 0)
            if player_id is not None and pid != player_id:
                continue
            
            # Get center position
            center = player.get("center", {})
            x = center.get("x")
            y = center.get("y")
            
            if x is None or y is None or x <= 0 or y <= 0:
                continue
            
            # Convert to int
            cx, cy = int(x), int(y)
            
            # Add Gaussian to heatmap at position
            half_k = kernel_size // 2
            y1 = max(0, cy - half_k)
            y2 = min(video_height, cy + half_k + 1)
            x1 = max(0, cx - half_k)
            x2 = min(video_width, cx + half_k + 1)
            
            ky1 = half_k - (cy - y1)
            ky2 = half_k + (y2 - cy)
            kx1 = half_k - (cx - x1)
            kx2 = half_k + (x2 - cx)
            
            if y2 > y1 and x2 > x1 and ky2 > ky1 and kx2 > kx1:
                heatmap[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    colormap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    cv_colormap = colormap_map.get(colormap, cv2.COLORMAP_TURBO)
    
    # Convert to 8-bit and apply colormap
    heatmap_8bit = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_8bit, cv_colormap)
    
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored, heatmap


def extract_video_frame(video_bytes: bytes, frame_number: Optional[int] = None) -> Optional[Any]:
    """
    Extract a frame from video bytes.
    
    Args:
        video_bytes: Video file as bytes
        frame_number: Frame to extract (None = middle frame)
    
    Returns:
        Frame as numpy array (RGB) or None
    """
    import numpy as np
    import cv2
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_number is None:
            frame_number = total_frames // 2
        
        frame_number = min(max(0, frame_number), total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        os.unlink(temp_path)


# =============================================================================
# PDF GENERATION
# =============================================================================

def generate_pdf_report(
    analysis_result: Dict[str, Any],
    video_frame: Optional[Any],
    heatmap: Optional[Any],
    config: Dict[str, Any]
) -> bytes:
    """
    Generate a PDF report using ReportLab.
    
    Args:
        analysis_result: Analysis result dictionary
        video_frame: Optional video frame (numpy array RGB)
        heatmap: Optional heatmap (numpy array RGB)
        config: Export configuration
    
    Returns:
        PDF file as bytes
    """
    import numpy as np
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    # Colors
    theme_green = colors.Color(0.133, 0.773, 0.369)
    theme_black = colors.Color(0.1, 0.1, 0.1)
    theme_gray = colors.Color(0.4, 0.4, 0.4)
    
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=theme_green,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=theme_black,
        spaceBefore=15,
        spaceAfter=10
    )
    text_style = ParagraphStyle(
        'Text',
        parent=styles['Normal'],
        fontSize=10,
        textColor=theme_black
    )
    
    # Build story
    story = []
    
    # Title
    title = config.get("title", "Badminton Video Analysis Report")
    story.append(Paragraph(title, title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=theme_green))
    story.append(Spacer(1, 0.3*inch))
    
    # Video info section
    story.append(Paragraph("Video Information", heading_style))
    
    video_info = [
        ["Duration", f"{analysis_result.get('duration', 0):.1f} seconds"],
        ["FPS", f"{analysis_result.get('fps', 30):.1f}"],
        ["Total Frames", str(analysis_result.get('total_frames', 0))],
        ["Processed Frames", str(analysis_result.get('processed_frames', 0))],
        ["Resolution", f"{analysis_result.get('video_width', 0)}x{analysis_result.get('video_height', 0)}"],
    ]
    
    info_table = Table(video_info, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), theme_black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Heatmap visualization
    if heatmap is not None and video_frame is not None:
        import cv2
        
        story.append(Paragraph("Position Heatmap", heading_style))
        
        # Blend heatmap with frame
        alpha = config.get("heatmap_alpha", 0.6)
        blended = cv2.addWeighted(video_frame, 1 - alpha, heatmap, alpha, 0)
        
        # Convert to PIL Image for ReportLab
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(blended)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Calculate size (fit to page width)
        max_width = 9 * inch
        aspect = video_frame.shape[1] / video_frame.shape[0]
        img_width = min(max_width, 6 * inch)
        img_height = img_width / aspect
        
        # Add image
        img = Image(img_buffer, width=img_width, height=img_height)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
    
    # Player statistics
    players = analysis_result.get("players", [])
    if players:
        story.append(Paragraph("Player Statistics", heading_style))
        
        player_data = [["Player", "Total Distance", "Avg Speed", "Max Speed"]]
        for p in players:
            player_data.append([
                f"Player {p.get('player_id', 0) + 1}",
                f"{p.get('total_distance', 0):.1f} m",
                f"{p.get('avg_speed', 0):.1f} km/h",
                f"{p.get('max_speed', 0):.1f} km/h",
            ])
        
        player_table = Table(player_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        player_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), theme_green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), theme_black),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, theme_gray),
        ]))
        story.append(player_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Shuttle statistics
    shuttle = analysis_result.get("shuttle")
    if shuttle:
        story.append(Paragraph("Shuttle Statistics", heading_style))
        
        shuttle_data = [
            ["Metric", "Value"],
            ["Shots Detected", str(shuttle.get("shots_detected", 0))],
            ["Average Speed", f"{shuttle.get('avg_speed', 0):.1f} km/h"],
            ["Maximum Speed", f"{shuttle.get('max_speed', 0):.1f} km/h"],
        ]
        
        shuttle_table = Table(shuttle_data, colWidths=[2*inch, 2*inch])
        shuttle_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), theme_green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), theme_black),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, theme_gray),
        ]))
        story.append(shuttle_table)
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=theme_gray))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=theme_gray,
        alignment=TA_CENTER,
        spaceBefore=10
    )
    story.append(Paragraph(
        f"Generated by Badminton Tracker | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


# =============================================================================
# MODAL ENDPOINT
# =============================================================================

@app.function(
    timeout=300,  # 5 minute timeout
    memory=2048,  # 2GB memory
)
@modal.fastapi_endpoint(method="POST")
async def generate_pdf(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a PDF report from video analysis data.
    
    Request body:
        videoUrl: URL to download video from Convex storage
        resultsUrl: URL to download analysis results from Convex storage
        config: PDF export configuration
            - title: Report title
            - heatmap_colormap: Colormap for heatmap (turbo, jet, hot, etc.)
            - heatmap_alpha: Heatmap overlay alpha (0-1)
            - frame_number: Frame to use for visualization
            - include_heatmap: Whether to include heatmap
    
    Returns:
        pdfBase64: Base64 encoded PDF file
        success: Boolean indicating success
        error: Error message if failed
    """
    import httpx
    import base64
    import numpy as np
    
    try:
        video_url = request.get("videoUrl")
        results_url = request.get("resultsUrl")
        config = request.get("config", {})
        
        if not video_url or not results_url:
            return {
                "success": False,
                "error": "Missing videoUrl or resultsUrl"
            }
        
        print(f"[PDF EXPORT] Downloading video from: {video_url[:80]}...")
        print(f"[PDF EXPORT] Downloading results from: {results_url[:80]}...")
        
        async with httpx.AsyncClient(timeout=120) as client:
            # Download video
            video_response = await client.get(video_url)
            if video_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to download video: {video_response.status_code}"
                }
            video_bytes = video_response.content
            print(f"[PDF EXPORT] Downloaded video: {len(video_bytes)} bytes")
            
            # Download results
            results_response = await client.get(results_url)
            if results_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to download results: {results_response.status_code}"
                }
            
            try:
                analysis_result = results_response.json()
            except:
                analysis_result = json.loads(results_response.text)
            print(f"[PDF EXPORT] Downloaded analysis results")
        
        # Extract frame
        video_frame = None
        if config.get("include_heatmap", True):
            frame_number = config.get("frame_number")
            video_frame = extract_video_frame(video_bytes, frame_number)
            if video_frame is not None:
                print(f"[PDF EXPORT] Extracted video frame: {video_frame.shape}")
        
        # Generate heatmap
        heatmap = None
        if video_frame is not None:
            skeleton_data = analysis_result.get("skeleton_data", [])
            video_width = analysis_result.get("video_width", video_frame.shape[1])
            video_height = analysis_result.get("video_height", video_frame.shape[0])
            colormap = config.get("heatmap_colormap", "turbo")
            
            heatmap, _ = generate_heatmap_overlay(
                skeleton_data, video_width, video_height,
                player_id=None, colormap=colormap
            )
            
            # Resize heatmap to match frame if needed
            import cv2
            if heatmap.shape[:2] != video_frame.shape[:2]:
                heatmap = cv2.resize(heatmap, (video_frame.shape[1], video_frame.shape[0]))
            
            print(f"[PDF EXPORT] Generated heatmap: {heatmap.shape}")
        
        # Generate PDF
        print(f"[PDF EXPORT] Generating PDF report...")
        pdf_bytes = generate_pdf_report(analysis_result, video_frame, heatmap, config)
        print(f"[PDF EXPORT] PDF generated: {len(pdf_bytes)} bytes")
        
        # Encode as base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return {
            "success": True,
            "pdfBase64": pdf_base64,
            "size": len(pdf_bytes)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[PDF EXPORT] Error: {error_msg}")
        print(traceback.format_exc())
        return {
            "success": False,
            "error": error_msg
        }


@app.function()
@modal.fastapi_endpoint(method="GET")
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "badminton-tracker-pdf-export",
        "timestamp": datetime.now().isoformat()
    }
