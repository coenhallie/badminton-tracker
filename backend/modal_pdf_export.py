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

# Player colors for heatmaps (RGB)
PLAYER_COLORS = [
    (230, 64, 64),   # Red
    (38, 140, 133),  # Cyan
    (38, 128, 153),  # Blue
    (89, 153, 115),  # Green
]

# Standard badminton court dimensions (meters)
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
SINGLES_WIDTH = 5.18
SERVICE_LINE = 1.98
BACK_SERVICE_LINE = 0.76


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


# =============================================================================
# MINIMAP GENERATION (Court diagram with movement trails)
# =============================================================================

def compute_homography(court_corners: List[List[float]]) -> Optional[Any]:
    """
    Compute a homography matrix from video pixel corners to standard court
    coordinates in meters.

    Supports 4-point (outer corners) or 12-point court keypoint arrays.
    Returns a 3×3 numpy homography matrix, or None on failure.
    """
    import numpy as np

    if not court_corners or len(court_corners) < 4:
        return None

    # Standard court positions in meters matching the 12-point keypoint system
    COURT_KEYPOINT_POSITIONS = [
        [0, 0],                                           # 0: TL
        [COURT_WIDTH, 0],                                 # 1: TR
        [COURT_WIDTH, COURT_LENGTH],                      # 2: BR
        [0, COURT_LENGTH],                                # 3: BL
        [0, COURT_LENGTH / 2],                            # 4: NL
        [COURT_WIDTH, COURT_LENGTH / 2],                  # 5: NR
        [0, COURT_LENGTH / 2 - SERVICE_LINE],             # 6: SNL
        [COURT_WIDTH, COURT_LENGTH / 2 - SERVICE_LINE],   # 7: SNR
        [0, COURT_LENGTH / 2 + SERVICE_LINE],             # 8: SFL
        [COURT_WIDTH, COURT_LENGTH / 2 + SERVICE_LINE],   # 9: SFR
        [COURT_WIDTH / 2, COURT_LENGTH / 2 - SERVICE_LINE],  # 10: CTN
        [COURT_WIDTH / 2, COURT_LENGTH / 2 + SERVICE_LINE],  # 11: CTF
    ]

    n = len(court_corners)
    src = np.array(court_corners[:min(n, 12)], dtype=np.float64)
    dst = np.array(COURT_KEYPOINT_POSITIONS[:min(n, 12)], dtype=np.float64)

    try:
        import cv2
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if n > 4 else 0)
        return H
    except Exception as e:
        print(f"[Minimap] Homography computation failed: {e}")
        return None


def get_feet_position(player: Dict) -> Optional[Tuple[float, float]]:
    """
    Extract feet position from a player dict (ankle midpoint, fallback to center).
    Mirrors MiniCourt.vue's getFeetPosition logic.
    """
    CONFIDENCE_THRESHOLD = 0.3
    keypoints = player.get("keypoints", [])
    if keypoints and len(keypoints) >= 17:
        # COCO indices: 15=left ankle, 16=right ankle
        la = keypoints[15] if len(keypoints) > 15 else None
        ra = keypoints[16] if len(keypoints) > 16 else None
        if (la and ra
                and la.get("x") is not None and la.get("y") is not None
                and ra.get("x") is not None and ra.get("y") is not None
                and la.get("confidence", 0) > CONFIDENCE_THRESHOLD
                and ra.get("confidence", 0) > CONFIDENCE_THRESHOLD):
            return ((la["x"] + ra["x"]) / 2, (la["y"] + ra["y"]) / 2)
        # Single ankle fallbacks
        for ankle in (la, ra):
            if (ankle and ankle.get("x") is not None and ankle.get("y") is not None
                    and ankle.get("confidence", 0) > CONFIDENCE_THRESHOLD):
                return (ankle["x"], ankle["y"])
        # Knee fallback (indices 13, 14)
        lk = keypoints[13] if len(keypoints) > 13 else None
        rk = keypoints[14] if len(keypoints) > 14 else None
        if (lk and rk
                and lk.get("x") is not None and lk.get("y") is not None
                and rk.get("x") is not None and rk.get("y") is not None
                and lk.get("confidence", 0) > CONFIDENCE_THRESHOLD
                and rk.get("confidence", 0) > CONFIDENCE_THRESHOLD):
            return ((lk["x"] + rk["x"]) / 2, (lk["y"] + rk["y"]) / 2)

    # Fallback: player centre
    center = player.get("center", {})
    if center.get("x") is not None and center.get("y") is not None:
        return (center["x"], center["y"])
    return None


def generate_minimap_image(
    skeleton_data: List[Dict],
    court_corners: Optional[List[List[float]]],
    img_width: int = 300,
    img_height: int = 550,
) -> Optional[Any]:
    """
    Render a top-down badminton court image with complete movement trails
    for every player across the full video.

    Args:
        skeleton_data:  List of frame dicts with ``players`` arrays.
        court_corners:  Court keypoints in video-pixel space (4 or 12 pts).
        img_width:      Output image width in pixels.
        img_height:     Output image height in pixels.

    Returns:
        RGB numpy array of the minimap, or None if court data is missing.
    """
    import numpy as np
    import cv2

    if not court_corners or len(court_corners) < 4:
        print("[Minimap] No court corners – skipping minimap")
        return None

    H = compute_homography(court_corners)
    if H is None:
        return None

    # ── layout constants ─────────────────────────────────────────────────
    padding = 20
    avail_w = img_width - 2 * padding
    avail_h = img_height - 2 * padding
    scale_x = avail_w / COURT_WIDTH
    scale_y = avail_h / COURT_LENGTH
    scale = min(scale_x, scale_y)
    court_w_px = int(COURT_WIDTH * scale)
    court_h_px = int(COURT_LENGTH * scale)
    off_x = (img_width - court_w_px) // 2
    off_y = (img_height - court_h_px) // 2

    def court_to_px(cx: float, cy: float) -> Tuple[int, int]:
        return (int(off_x + cx * scale), int(off_y + cy * scale))

    # ── draw court ───────────────────────────────────────────────────────
    img = np.full((img_height, img_width, 3), (17, 17, 17), dtype=np.uint8)  # dark bg
    # Court surface
    cv2.rectangle(img, (off_x, off_y), (off_x + court_w_px, off_y + court_h_px),
                  (26, 71, 42), -1)  # dark green fill
    # Outer boundary
    cv2.rectangle(img, (off_x, off_y), (off_x + court_w_px, off_y + court_h_px),
                  (255, 255, 255), 2)
    # Singles sidelines
    singles_off = int((COURT_WIDTH - SINGLES_WIDTH) / 2 * scale)
    cv2.line(img, (off_x + singles_off, off_y),
             (off_x + singles_off, off_y + court_h_px), (255, 255, 255), 1)
    cv2.line(img, (off_x + court_w_px - singles_off, off_y),
             (off_x + court_w_px - singles_off, off_y + court_h_px), (255, 255, 255), 1)
    # Net
    net_y = off_y + court_h_px // 2
    cv2.line(img, (off_x, net_y), (off_x + court_w_px, net_y), (68, 68, 255), 3)
    # Service lines
    svc = int(SERVICE_LINE * scale)
    cv2.line(img, (off_x, net_y - svc), (off_x + court_w_px, net_y - svc), (255, 255, 255), 1)
    cv2.line(img, (off_x, net_y + svc), (off_x + court_w_px, net_y + svc), (255, 255, 255), 1)
    # Centre lines in service boxes
    cx_px = off_x + court_w_px // 2
    cv2.line(img, (cx_px, off_y), (cx_px, net_y - svc), (255, 255, 255), 1)
    cv2.line(img, (cx_px, net_y + svc), (cx_px, off_y + court_h_px), (255, 255, 255), 1)

    # ── collect trails ───────────────────────────────────────────────────
    # Dict mapping player_id → list of (court_x, court_y) in metres
    trails: Dict[int, List[Tuple[float, float]]] = {}
    margin = 2.0  # metres outside court to still accept

    for frame_data in skeleton_data:
        for player in frame_data.get("players", []):
            pid = player.get("player_id", 0)
            feet = get_feet_position(player)
            if feet is None:
                continue
            # Transform to court metres via homography
            pt = np.array([[[feet[0], feet[1]]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(pt, H)
            cx_m, cy_m = float(dst[0][0][0]), float(dst[0][0][1])
            # Reject out-of-bounds points (bad detections)
            if cx_m < -margin or cx_m > COURT_WIDTH + margin:
                continue
            if cy_m < -margin or cy_m > COURT_LENGTH + margin:
                continue
            trails.setdefault(pid, []).append((cx_m, cy_m))

    if not trails:
        print("[Minimap] No valid trail data – skipping minimap")
        return None

    # ── draw trails ──────────────────────────────────────────────────────
    # Player RGB colours (matching frontend PLAYER_COLORS)
    TRAIL_COLORS_BGR = [
        (64, 64, 230),    # Red   → BGR
        (133, 140, 38),   # Cyan  → BGR
        (153, 128, 38),   # Blue  → BGR
        (115, 153, 89),   # Green → BGR
    ]

    for pid, points in sorted(trails.items()):
        color = TRAIL_COLORS_BGR[pid % len(TRAIL_COLORS_BGR)]
        # Build polyline pixel points
        pts_px = [court_to_px(cx, cy) for cx, cy in points]

        # Draw the trail as a polyline (anti-aliased)
        if len(pts_px) >= 2:
            pts_arr = np.array(pts_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_arr], isClosed=False, color=color,
                          thickness=2, lineType=cv2.LINE_AA)

    # ── draw legend below court ──────────────────────────────────────────
    legend_y = off_y + court_h_px + 12
    x_cursor = off_x
    for pid in sorted(trails.keys()):
        color_bgr = TRAIL_COLORS_BGR[pid % len(TRAIL_COLORS_BGR)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        cv2.circle(img, (x_cursor + 6, legend_y + 6), 6, color_bgr, -1, cv2.LINE_AA)
        cv2.putText(img, f"P{pid + 1}", (x_cursor + 16, legend_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        x_cursor += 55

    # Convert BGR → RGB for PIL / ReportLab
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


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
    config: Dict[str, Any],
    minimap: Optional[Any] = None,
) -> bytes:
    """
    Generate a PDF report using ReportLab.
    
    Args:
        analysis_result: Analysis result dictionary
        video_frame: Optional video frame (numpy array RGB)
        heatmap: Optional heatmap (numpy array RGB)
        config: Export configuration
        minimap: Optional minimap image (numpy array RGB) - court with movement trails
    
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
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Normal'],
        fontSize=10,
        textColor=theme_gray,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=4,
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
    
    # Heatmap + Minimap visualization (side by side)
    has_heatmap = heatmap is not None and video_frame is not None
    has_minimap = minimap is not None
    
    if has_heatmap or has_minimap:
        import cv2
        from PIL import Image as PILImage
        
        story.append(Paragraph("Position Heatmap &amp; Court Movement", heading_style))
        
        vis_cells = []   # cells for the side-by-side table row
        vis_widths = []   # column widths
        
        # --- Heatmap cell ---
        if has_heatmap:
            alpha = config.get("heatmap_alpha", 0.6)
            blended = cv2.addWeighted(video_frame, 1 - alpha, heatmap, alpha, 0)
            pil_hm = PILImage.fromarray(blended)
            hm_buf = io.BytesIO()
            pil_hm.save(hm_buf, format='PNG')
            hm_buf.seek(0)
            
            # Size the heatmap – leave room for minimap if present
            aspect = video_frame.shape[1] / video_frame.shape[0]
            if has_minimap:
                hm_w = 5.8 * inch
            else:
                hm_w = min(9 * inch, 6 * inch)
            hm_h = hm_w / aspect
            
            hm_img = Image(hm_buf, width=hm_w, height=hm_h)
            # Wrap image + subtitle in a mini table for vertical stacking
            hm_cell_content = Table(
                [[hm_img], [Paragraph("Position Heatmap", subheading_style)]],
                colWidths=[hm_w],
            )
            hm_cell_content.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            vis_cells.append(hm_cell_content)
            vis_widths.append(hm_w + 0.1 * inch)
        
        # --- Minimap cell ---
        if has_minimap:
            pil_mm = PILImage.fromarray(minimap)
            mm_buf = io.BytesIO()
            pil_mm.save(mm_buf, format='PNG')
            mm_buf.seek(0)
            
            mm_aspect = minimap.shape[1] / minimap.shape[0]
            # Match the target height to the heatmap height if both are present
            if has_heatmap:
                mm_h = hm_h  # type: ignore[possibly-undefined]
            else:
                mm_h = 4.0 * inch
            mm_w = mm_h * mm_aspect
            # Clamp minimap width
            mm_w = min(mm_w, 3.5 * inch)
            mm_h = mm_w / mm_aspect
            
            mm_img = Image(mm_buf, width=mm_w, height=mm_h)
            mm_cell_content = Table(
                [[mm_img], [Paragraph("Court Movement Map", subheading_style)]],
                colWidths=[mm_w],
            )
            mm_cell_content.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            vis_cells.append(mm_cell_content)
            vis_widths.append(mm_w + 0.1 * inch)
        
        # Create a single-row table to hold them side by side
        if vis_cells:
            vis_table = Table([vis_cells], colWidths=vis_widths)
            vis_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]))
            story.append(vis_table)
            story.append(Spacer(1, 0.3 * inch))
    
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
            
            # Merge frontend player data into analysis_result if provided.
            # This ensures the PDF shows the same calibrated speed values
            # as the dashboard (which may have recalculated speeds with
            # court keypoints and proper filtering).
            if config.get("players"):
                frontend_players = config["players"]
                print(f"[PDF EXPORT] Using frontend player data ({len(frontend_players)} players) for consistency with dashboard")
                analysis_result["players"] = frontend_players
            
            # Also merge other frontend data if provided
            for key in ("duration", "fps", "total_frames", "processed_frames",
                         "video_width", "video_height"):
                if config.get(key) is not None:
                    analysis_result[key] = config[key]
            
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
        
        # Generate minimap (court diagram with movement trails)
        minimap = None
        skeleton_data = analysis_result.get("skeleton_data", [])
        # Court corners: check multiple locations
        # 1) top-level config key  2) config.court_detection  3) results.court_detection
        court_corners = config.get("court_corners")
        if not court_corners:
            config_court = config.get("court_detection") or {}
            court_corners = config_court.get("court_corners")
        if not court_corners:
            result_court = analysis_result.get("court_detection") or {}
            court_corners = result_court.get("court_corners")
        
        if court_corners and skeleton_data:
            try:
                minimap = generate_minimap_image(skeleton_data, court_corners)
                if minimap is not None:
                    print(f"[PDF EXPORT] Generated minimap: {minimap.shape}")
                else:
                    print("[PDF EXPORT] Minimap generation returned None (insufficient data)")
            except Exception as e:
                print(f"[PDF EXPORT] Minimap generation failed (non-fatal): {e}")
        else:
            print(f"[PDF EXPORT] Skipping minimap: court_corners={'yes' if court_corners else 'no'}, skeleton_frames={len(skeleton_data)}")
        
        # Generate PDF
        print(f"[PDF EXPORT] Generating PDF report...")
        pdf_bytes = generate_pdf_report(analysis_result, video_frame, heatmap, config, minimap)
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
