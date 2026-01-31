"""
PDF Export Module for Badminton Video Analysis

Generates professional PDF reports containing:
- Video frame with heatmap overlay
- Player statistics
- Shuttle/shot analytics
- Court detection info
- Speed analytics

Uses ReportLab for PDF generation and OpenCV for image processing.
"""

import io
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Import heatmap generator
from heatmap_generator import (
    HeatmapGenerator, HeatmapConfig, generate_heatmap_from_skeleton_data,
    COLORMAP_MAPPING
)

# Import shuttle analytics for zone recalculation
from shuttle_analytics import recalculate_zone_analytics_from_skeleton_data


# =============================================================================
# CONSTANTS
# =============================================================================

# Colors for PDF (light theme - better for printing/PDF rendering)
THEME_GREEN = colors.Color(0.133, 0.773, 0.369)  # #22c55e
THEME_DARK_GREEN = colors.Color(0.086, 0.639, 0.290)  # Darker green for text
THEME_BLACK = colors.Color(0.1, 0.1, 0.1)  # Near black for main text
THEME_GRAY = colors.Color(0.4, 0.4, 0.4)  # #666666 - secondary text
THEME_LIGHT_GRAY = colors.Color(0.95, 0.95, 0.95)  # Light background
THEME_WHITE = colors.Color(1.0, 1.0, 1.0)

# Player colors (darker variants for better PDF visibility)
PLAYER_COLORS = [
    colors.Color(0.9, 0.25, 0.25),  # Dark Red
    colors.Color(0.15, 0.55, 0.52),  # Dark Cyan
    colors.Color(0.15, 0.50, 0.60),  # Dark Blue
    colors.Color(0.35, 0.60, 0.45),  # Dark Green
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PDFExportConfig:
    """Configuration for PDF export"""
    include_heatmap: bool = True
    heatmap_colormap: str = "turbo"
    heatmap_alpha: float = 0.6
    frame_number: Optional[int] = None  # None = use middle frame
    include_player_stats: bool = True
    include_shuttle_stats: bool = True
    include_court_info: bool = True
    include_speed_stats: bool = True
    page_size: Tuple = landscape(A4)
    title: str = "Badminton Video Analysis Report"


# =============================================================================
# PDF REPORT GENERATOR CLASS
# =============================================================================

class PDFReportGenerator:
    """
    Generates professional PDF reports for badminton video analysis.
    
    Usage:
        generator = PDFReportGenerator(analysis_result, video_path)
        pdf_bytes = generator.generate()
    """
    
    def __init__(
        self,
        analysis_result: Dict[str, Any],
        video_path: Path,
        config: Optional[PDFExportConfig] = None
    ):
        """
        Initialize the PDF report generator.
        
        Args:
            analysis_result: Complete analysis result dictionary from backend
            video_path: Path to the original video file
            config: Optional export configuration
        """
        self.result = analysis_result
        self.video_path = Path(video_path)
        self.config = config or PDFExportConfig()
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Pre-extract data
        self.video_id = analysis_result.get("video_id", "Unknown")
        self.duration = analysis_result.get("duration", 0)
        self.fps = analysis_result.get("fps", 30)
        self.total_frames = analysis_result.get("total_frames", 0)
        self.processed_frames = analysis_result.get("processed_frames", 0)
        self.video_width = analysis_result.get("video_width", 1920)
        self.video_height = analysis_result.get("video_height", 1080)
        self.players = analysis_result.get("players", [])
        self.shuttle = analysis_result.get("shuttle")
        self.skeleton_data = analysis_result.get("skeleton_data", [])
        self.court_detection = analysis_result.get("court_detection")
        self.shuttle_analytics = analysis_result.get("shuttle_analytics")
        
        # Get stored player zone analytics
        stored_zone_analytics = analysis_result.get("player_zone_analytics")
        
        # CRITICAL FIX: Recalculate zone analytics using manual keypoints if available
        # The stored zone analytics may be 0% if manual keypoints weren't set during
        # initial video processing. We need to recalculate from skeleton data.
        self.player_zone_analytics = self._get_recalculated_zone_analytics(stored_zone_analytics)
    
    def _get_recalculated_zone_analytics(
        self,
        stored_analytics: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get zone analytics, recalculating if necessary using manual keypoints.
        
        This fixes the issue where zone coverage is always 0% in PDF exports
        because manual keypoints were set after video processing.
        
        Args:
            stored_analytics: The originally stored zone analytics from video processing
            
        Returns:
            Zone analytics - either recalculated or original
        """
        # Check if we have skeleton data to recalculate from
        if not self.skeleton_data:
            print("[PDF EXPORT] No skeleton data available for zone recalculation")
            return stored_analytics
        
        # Check if stored analytics have any non-zero zone coverage
        has_valid_stored_analytics = False
        if stored_analytics:
            for player_id, analytics in stored_analytics.items():
                zone_coverage = analytics.get("zone_coverage", {})
                total = sum(zone_coverage.get(k, 0) for k in ["front", "mid", "back"])
                if total > 0:
                    has_valid_stored_analytics = True
                    break
        
        if has_valid_stored_analytics:
            print("[PDF EXPORT] Using stored zone analytics (already has valid data)")
            return stored_analytics
        
        # Stored analytics are all 0% - try to recalculate using manual keypoints
        print("[PDF EXPORT] Stored zone analytics are 0% - attempting recalculation with manual keypoints")
        
        try:
            recalculated = recalculate_zone_analytics_from_skeleton_data(
                skeleton_frames=self.skeleton_data,
                video_width=self.video_width,
                video_height=self.video_height
            )
            
            if recalculated:
                # Verify recalculated analytics have non-zero values
                for player_id, analytics in recalculated.items():
                    zone_coverage = analytics.get("zone_coverage", {})
                    total = sum(zone_coverage.get(k, 0) for k in ["front", "mid", "back"])
                    if total > 0:
                        print(f"[PDF EXPORT] Successfully recalculated zone analytics with manual keypoints")
                        return recalculated
                
                print("[PDF EXPORT] Recalculated analytics still show 0% - manual keypoints may not be set")
            else:
                print("[PDF EXPORT] Zone recalculation returned empty - manual keypoints may not be set")
        except Exception as e:
            print(f"[PDF EXPORT] Zone recalculation failed: {e}")
        
        # Fall back to stored analytics
        return stored_analytics
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report - light theme for PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=THEME_BLACK,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=THEME_DARK_GREEN,
            spaceAfter=12,
            spaceBefore=16,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=THEME_BLACK,
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='ReportText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=THEME_BLACK,
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        # Stats value style
        self.styles.add(ParagraphStyle(
            name='StatValue',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=THEME_DARK_GREEN,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Stats label style
        self.styles.add(ParagraphStyle(
            name='StatLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=THEME_GRAY,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
    
    def _extract_video_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract a frame from the video.
        
        Args:
            frame_number: Specific frame to extract. If None, uses middle frame.
            
        Returns:
            BGR frame as numpy array, or None on failure
        """
        if not self.video_path.exists():
            print(f"[PDF Export] Video file not found: {self.video_path}")
            return None
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[PDF Export] Failed to open video: {self.video_path}")
            return None
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine which frame to extract
            if frame_number is not None:
                target_frame = min(frame_number, total_frames - 1)
            else:
                # Use middle frame
                target_frame = total_frames // 2
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if not ret:
                print(f"[PDF Export] Failed to read frame {target_frame}")
                return None
            
            return frame
        finally:
            cap.release()
    
    def _generate_heatmap_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Generate heatmap overlay on frame.
        
        Args:
            frame: BGR video frame
            alpha: Heatmap overlay opacity (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        if not self.skeleton_data:
            return frame
        
        # Create heatmap config
        config = HeatmapConfig(
            colormap=self.config.heatmap_colormap,
            intensity_scale=1.0,
            decay_rate=1.0  # No decay for static heatmap
        )
        
        # Generate heatmap from all skeleton data
        heatmap_data = generate_heatmap_from_skeleton_data(
            skeleton_frames=self.skeleton_data,
            video_width=self.video_width,
            video_height=self.video_height,
            video_id=self.video_id,
            config=config
        )
        
        # Get the combined heatmap
        heatmap = heatmap_data.combined_heatmap
        if heatmap is None or heatmap.max() == 0:
            return frame
        
        # Normalize heatmap to 0-255
        normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Resize to frame dimensions
        frame_h, frame_w = frame.shape[:2]
        if normalized.shape != (frame_h, frame_w):
            normalized = cv2.resize(normalized, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        colormap = COLORMAP_MAPPING.get(self.config.heatmap_colormap, cv2.COLORMAP_TURBO)
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Create mask where heatmap has values
        mask = normalized > 0
        
        # Blend with frame
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
        
        return result
    
    def _frame_to_reportlab_image(
        self,
        frame: np.ndarray,
        max_width: float = 9 * inch,
        max_height: float = 5 * inch
    ) -> Image:
        """
        Convert OpenCV frame to ReportLab Image object.
        
        Args:
            frame: BGR frame
            max_width: Maximum width in reportlab units
            max_height: Maximum height in reportlab units
            
        Returns:
            ReportLab Image object
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode to PNG in memory
        is_success, buffer = cv2.imencode('.png', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise ValueError("Failed to encode frame to PNG")
        
        # Create BytesIO object
        img_buffer = io.BytesIO(buffer.tobytes())
        
        # Calculate aspect ratio preserving dimensions
        frame_h, frame_w = frame.shape[:2]
        aspect_ratio = frame_w / frame_h
        
        if aspect_ratio > (max_width / max_height):
            # Width constrained
            width = max_width
            height = max_width / aspect_ratio
        else:
            # Height constrained
            height = max_height
            width = max_height * aspect_ratio
        
        return Image(img_buffer, width=width, height=height)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    def _format_distance(self, meters: float) -> str:
        """Format distance with appropriate units"""
        if meters >= 1000:
            return f"{meters / 1000:.2f} km"
        return f"{meters:.1f} m"
    
    def _format_speed(self, kmh: float) -> str:
        """Format speed"""
        return f"{kmh:.1f} km/h"
    
    def _build_summary_section(self) -> List:
        """Build the video summary section"""
        elements = []
        
        elements.append(Paragraph("Video Summary", self.styles['SectionHeader']))
        
        # Summary table
        summary_data = [
            ["Duration", "Frames Analyzed", "Players Detected", "FPS"],
            [
                self._format_duration(self.duration),
                str(self.processed_frames),
                str(len(self.players)),
                f"{self.fps:.0f}"
            ]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.2*inch]*4)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
            ('BACKGROUND', (0, 1), (-1, 1), THEME_WHITE),
            ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
            ('TEXTCOLOR', (0, 1), (-1, 1), THEME_BLACK),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_movement_section(self) -> List:
        """Build the movement analysis section"""
        elements = []
        
        elements.append(Paragraph("Movement Analysis", self.styles['SectionHeader']))
        
        # Calculate totals
        total_distance = sum(p.get("total_distance", 0) for p in self.players)
        avg_speed = sum(p.get("avg_speed", 0) for p in self.players) / max(len(self.players), 1)
        max_speed = max((p.get("max_speed", 0) for p in self.players), default=0)
        
        # Movement stats table
        movement_data = [
            ["Total Distance", "Average Speed", "Max Speed"],
            [
                self._format_distance(total_distance),
                self._format_speed(avg_speed),
                self._format_speed(max_speed)
            ]
        ]
        
        movement_table = Table(movement_data, colWidths=[2.9*inch]*3)
        movement_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
            ('BACKGROUND', (0, 1), (-1, 1), THEME_WHITE),
            ('BACKGROUND', (0, 1), (0, 1), colors.Color(0.9, 0.98, 0.9)),  # Light green highlight
            ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
            ('TEXTCOLOR', (0, 1), (-1, 1), THEME_BLACK),
            ('TEXTCOLOR', (0, 1), (0, 1), THEME_DARK_GREEN),  # Dark green for total distance
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('BOX', (0, 1), (0, 1), 1, THEME_GREEN),  # Border around highlighted cell
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ]))
        
        elements.append(movement_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_player_stats_section(self) -> List:
        """Build the player statistics section"""
        elements = []
        
        if not self.players:
            return elements
        
        elements.append(Paragraph("Player Statistics", self.styles['SectionHeader']))
        
        # Player stats table header
        player_headers = ["Player", "Distance", "Avg Speed", "Max Speed", "Tracked Frames"]
        player_rows = [player_headers]
        
        for i, player in enumerate(self.players):
            player_id = player.get("player_id", i + 1)
            player_rows.append([
                f"Player {player_id}",
                self._format_distance(player.get("total_distance", 0)),
                self._format_speed(player.get("avg_speed", 0)),
                self._format_speed(player.get("max_speed", 0)),
                str(len(player.get("positions", [])))
            ])
        
        player_table = Table(player_rows, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
        
        # Dynamic styling for player rows with colors
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
            ('BACKGROUND', (0, 1), (-1, -1), THEME_WHITE),
            ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ]
        
        # Add player colors to first column
        for i in range(len(self.players)):
            row_idx = i + 1
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            table_style.append(('TEXTCOLOR', (0, row_idx), (0, row_idx), color))
            table_style.append(('TEXTCOLOR', (1, row_idx), (-1, row_idx), THEME_BLACK))
            # Left border with player color
            table_style.append(('LINEBEFORE', (0, row_idx), (0, row_idx), 3, color))
        
        player_table.setStyle(TableStyle(table_style))
        
        elements.append(player_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_shuttle_section(self) -> List:
        """Build the shuttle/shot analysis section"""
        elements = []
        
        if not self.shuttle and not self.shuttle_analytics:
            return elements
        
        elements.append(Paragraph("Shuttle Analytics", self.styles['SectionHeader']))
        
        if self.shuttle:
            shuttle_data = [
                ["Shots Detected", "Average Speed", "Max Speed"],
                [
                    str(self.shuttle.get("shots_detected", 0)),
                    self._format_speed(self.shuttle.get("avg_speed", 0)),
                    self._format_speed(self.shuttle.get("max_speed", 0))
                ]
            ]
            
            shuttle_table = Table(shuttle_data, colWidths=[2.9*inch]*3)
            shuttle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
                ('BACKGROUND', (0, 1), (-1, 1), THEME_WHITE),
                ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
                ('TEXTCOLOR', (0, 1), (-1, 1), THEME_DARK_GREEN),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, 1), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 1), (-1, 1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
            ]))
            
            elements.append(shuttle_table)
        
        # Enhanced shuttle analytics if available
        if self.shuttle_analytics:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph("Shot Type Distribution", self.styles['SubsectionHeader']))
            
            shot_types = self.shuttle_analytics.get("shot_types", {})
            if shot_types:
                shot_headers = ["Shot Type", "Count"]
                shot_rows = [shot_headers]
                for shot_type, count in shot_types.items():
                    if count > 0:
                        shot_rows.append([shot_type.replace("_", " ").title(), str(count)])
                
                if len(shot_rows) > 1:
                    shot_table = Table(shot_rows, colWidths=[3*inch, 2*inch])
                    shot_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
                        ('BACKGROUND', (0, 1), (-1, -1), THEME_WHITE),
                        ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
                        ('TEXTCOLOR', (0, 1), (-1, -1), THEME_BLACK),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
                    ]))
                    elements.append(shot_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_court_section(self) -> List:
        """Build the court detection section"""
        elements = []
        
        if not self.court_detection:
            return elements
        
        elements.append(Paragraph("Court Detection", self.styles['SectionHeader']))
        
        detected = self.court_detection.get("detected", False)
        confidence = self.court_detection.get("confidence", 0)
        court_dims = self.court_detection.get("court_dimensions", {})
        
        court_data = [
            ["Status", "Confidence", "Court Dimensions"],
            [
                "Detected ✓" if detected else "Not Detected",
                f"{confidence * 100:.1f}%",
                f"{court_dims.get('width_meters', 6.1):.1f}m × {court_dims.get('length_meters', 13.4):.1f}m"
            ]
        ]
        
        court_table = Table(court_data, colWidths=[2.9*inch]*3)
        court_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
            ('BACKGROUND', (0, 1), (-1, 1), THEME_WHITE),
            ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
            ('TEXTCOLOR', (0, 1), (-1, 1), THEME_DARK_GREEN if detected else THEME_GRAY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ]))
        
        elements.append(court_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_zone_analytics_section(self) -> List:
        """Build the player zone analytics section"""
        elements = []
        
        if not self.player_zone_analytics:
            return elements
        
        elements.append(Paragraph("Court Coverage Analysis", self.styles['SectionHeader']))
        
        for player_id_str, analytics in self.player_zone_analytics.items():
            player_id = int(player_id_str)
            color = PLAYER_COLORS[(player_id - 1) % len(PLAYER_COLORS)]
            
            elements.append(Paragraph(
                f"<font color='#{int(color.red*255):02x}{int(color.green*255):02x}{int(color.blue*255):02x}'>Player {player_id}</font>",
                self.styles['SubsectionHeader']
            ))
            
            zone_coverage = analytics.get("zone_coverage", {})
            
            # Zone coverage table
            zone_data = [
                ["Front", "Mid", "Back", "Left", "Center", "Right"],
                [
                    f"{zone_coverage.get('front', 0):.1f}%",
                    f"{zone_coverage.get('mid', 0):.1f}%",
                    f"{zone_coverage.get('back', 0):.1f}%",
                    f"{zone_coverage.get('left', 0):.1f}%",
                    f"{zone_coverage.get('center', 0):.1f}%",
                    f"{zone_coverage.get('right', 0):.1f}%"
                ]
            ]
            
            zone_table = Table(zone_data, colWidths=[1.4*inch]*6)
            zone_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), THEME_LIGHT_GRAY),
                ('BACKGROUND', (0, 1), (-1, 1), THEME_WHITE),
                ('TEXTCOLOR', (0, 0), (-1, 0), THEME_GRAY),
                ('TEXTCOLOR', (0, 1), (-1, 1), THEME_BLACK),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, 1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
            ]))
            
            elements.append(zone_table)
            
            # Avg distance to net
            avg_dist = analytics.get("avg_distance_to_net_m", 0)
            elements.append(Paragraph(
                f"Average distance to net: <b>{avg_dist:.2f}m</b>",
                self.styles['ReportText']
            ))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def generate(self) -> bytes:
        """
        Generate the complete PDF report.
        
        Returns:
            PDF file as bytes
        """
        # Create buffer for PDF
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.config.page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph(self.config.title, self.styles['ReportTitle']))
        story.append(Paragraph(
            f"Video ID: {self.video_id[:8]}... | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ParagraphStyle(
                name='Subtitle',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=THEME_GRAY,
                alignment=TA_CENTER
            )
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Heatmap visualization with frame
        if self.config.include_heatmap:
            story.append(Paragraph("Position Heatmap", self.styles['SectionHeader']))
            
            # Extract frame
            frame = self._extract_video_frame(self.config.frame_number)
            
            if frame is not None:
                # Apply heatmap overlay
                frame_with_heatmap = self._generate_heatmap_overlay(
                    frame,
                    alpha=self.config.heatmap_alpha
                )
                
                # Convert to ReportLab image
                img = self._frame_to_reportlab_image(frame_with_heatmap)
                story.append(img)
                
                # Caption
                frame_num = self.config.frame_number or (self.total_frames // 2)
                story.append(Paragraph(
                    f"<i>Frame {frame_num} with cumulative player position heatmap ({self.config.heatmap_colormap} colormap)</i>",
                    ParagraphStyle(
                        name='Caption',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        textColor=THEME_GRAY,
                        alignment=TA_CENTER,
                        spaceBefore=6
                    )
                ))
            else:
                story.append(Paragraph(
                    "Video frame could not be extracted",
                    self.styles['ReportText']
                ))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Summary section
        story.extend(self._build_summary_section())
        
        # Movement analysis
        if self.config.include_speed_stats:
            story.extend(self._build_movement_section())
        
        # Player statistics
        if self.config.include_player_stats:
            story.extend(self._build_player_stats_section())
        
        # Shuttle analytics
        if self.config.include_shuttle_stats:
            story.extend(self._build_shuttle_section())
        
        # Court detection
        if self.config.include_court_info:
            story.extend(self._build_court_section())
        
        # Zone analytics
        story.extend(self._build_zone_analytics_section())
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=THEME_GRAY))
        story.append(Paragraph(
            "Generated by Badminton Tracker | https://github.com/badminton-tracker",
            ParagraphStyle(
                name='Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=THEME_GRAY,
                alignment=TA_CENTER,
                spaceBefore=10
            )
        ))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_pdf_report(
    analysis_result: Dict[str, Any],
    video_path: Path,
    config: Optional[PDFExportConfig] = None
) -> bytes:
    """
    Convenience function to generate a PDF report.
    
    Args:
        analysis_result: Analysis result dictionary from backend
        video_path: Path to the original video file
        config: Optional export configuration
        
    Returns:
        PDF file as bytes
    """
    generator = PDFReportGenerator(analysis_result, video_path, config)
    return generator.generate()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PDFReportGenerator",
    "PDFExportConfig",
    "generate_pdf_report",
]
