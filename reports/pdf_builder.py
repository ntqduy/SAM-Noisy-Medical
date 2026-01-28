"""
PDF Report Builder for SAM Benchmark Results.

Generates comprehensive PDF reports including:
  - Executive summary with config snapshot
  - Dataset & phase settings
  - Noise preset tables (with intensity scalars)
  - Quantitative results tables (Dice, IoU, HD95)
  - Extended stability metrics (drop_Lmax, slope, AUC, CV)
  - Uncertainty analysis
  - Noise gallery visualizations
  - Sensitivity plots and global heatmaps
  - Side-by-side comparison grids
  - Failure case analysis
  - Appendix with raw data references

Extended for AIO25 NoisySAM project requirements.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import warnings

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, gray


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation safely, handling constant arrays.
    
    Returns:
        Correlation coefficient, or 0.0 if computation fails (e.g., zero stddev)
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    
    # Check for constant arrays (stddev = 0)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return corr
        except Exception:
            return 0.0
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib import colors


class PDFReportBuilder:
    """Build comprehensive PDF reports for benchmark results."""
    
    # Supported image extensions for RLImage
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    
    def __init__(self, cfg: dict, exp_dir: Path):
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _is_valid_image(self, path: str) -> bool:
        """Check if path is a valid image file (not PDF)."""
        p = Path(path)
        return p.suffix.lower() in self.IMAGE_EXTENSIONS and p.exists()
    
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#1a365d'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#2c5282'),
            spaceBefore=15,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#4a5568'),
            spaceBefore=10,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='MyBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=gray
        ))
    
    def build_report(
        self,
        df: pd.DataFrame,
        agg_df: pd.DataFrame,
        stability_df: pd.DataFrame,
        figure_paths: List[str],
        failure_paths: List[str],
        out_path: Path,
        noise_gallery_paths: List[str] = None,
        global_plot_paths: List[str] = None
    ):
        """
        Build the complete PDF report.
        
        Args:
            df: Per-sample results DataFrame
            agg_df: Aggregated results DataFrame
            stability_df: Stability metrics DataFrame
            figure_paths: List of figure image paths
            failure_paths: List of failure case image paths
            out_path: Output PDF path
            noise_gallery_paths: List of noise gallery image paths
            global_plot_paths: List of global comparative plot paths
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        
        # Title page
        story.extend(self._build_title_page())
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._build_executive_summary(df, agg_df, stability_df))
        story.append(PageBreak())
        
        # Dataset & Phase Settings
        story.extend(self._build_dataset_section())
        
        # Noise Preset Tables
        story.extend(self._build_noise_preset_section())
        story.append(PageBreak())
        
        # Quantitative Results
        story.extend(self._build_quantitative_section(agg_df))
        
        # Per-noise-type metrics by level (NEW)
        story.extend(self._build_per_noise_metrics_section(df))
        story.append(PageBreak())
        
        # Stability Analysis (extended)
        story.extend(self._build_stability_section_extended(stability_df))
        
        # Uncertainty Analysis
        story.extend(self._build_uncertainty_section(df))
        
        # Noise Gallery
        if noise_gallery_paths:
            story.extend(self._build_noise_gallery_section(noise_gallery_paths))
        
        # Global Comparative Plots
        if global_plot_paths:
            story.extend(self._build_global_plots_section(global_plot_paths))
        
        # Mode Comparison Plots (automatic vs prompt_bbox)
        story.extend(self._build_mode_comparison_section(figure_paths))
        
        # Sensitivity Plots
        story.extend(self._build_plots_section(figure_paths))
        
        # Failure Cases
        story.extend(self._build_failure_section(failure_paths))
        
        # Appendix
        story.extend(self._build_appendix())
        
        doc.build(story)
    
    def _build_title_page(self) -> List:
        """Build title page elements."""
        story = []
        
        story.append(Spacer(1, 3*cm))
        
        title = Paragraph(
            "SAM Benchmark Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        
        story.append(Spacer(1, 0.5*cm))
        
        subtitle = Paragraph(
            "Noisy Medical Radiology Imaging Conditions",
            self.styles['SectionTitle']
        )
        story.append(subtitle)
        
        story.append(Spacer(1, 2*cm))
        
        exp_name = self.cfg.get("exp", {}).get("name", "Unknown")
        phase = self.cfg.get("phase", 1)
        
        info_data = [
            ["Experiment Name:", exp_name],
            ["Phase:", str(phase)],
            # ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        
        info_table = Table(info_data, colWidths=[4*cm, 8*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        
        return story
    
    def _build_executive_summary(self, df: pd.DataFrame, agg_df: pd.DataFrame, stability_df: pd.DataFrame) -> List:
        """Build executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        
        # Config snapshot
        story.append(Paragraph("Configuration Snapshot", self.styles['SubSection']))
        
        config_items = [
            f"• Phase: {self.cfg.get('phase', 1)}",
            f"• Device: {self.cfg.get('device', 'cpu')}",
            f"• Seed: {self.cfg.get('seed', 42)}",
            f"• Datasets: {len(self.cfg.get('datasets', []))}",
            f"• Models: {len(self.cfg.get('models', []))}",
        ]
        for item in config_items:
            story.append(Paragraph(item, self.styles['MyBodyText']))
        
        story.append(Spacer(1, 0.5*cm))
        
        # Summary statistics
        story.append(Paragraph("Summary Statistics", self.styles['SubSection']))
        
        if len(df) > 0:
            n_samples = df["id"].nunique()
            n_datasets = df["dataset"].nunique()
            n_models = df[["model", "weight"]].drop_duplicates().shape[0]
            
            baseline = df[df["protocol"] == "P0"]
            mean_dice_clean = baseline["dice"].mean() if len(baseline) > 0 else 0
            
            worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")]
            mean_dice_l4 = worst["dice"].mean() if len(worst) > 0 else 0
            
            summary_data = [
                ["Metric", "Value"],
                ["Total Samples", str(n_samples)],
                ["Datasets", str(n_datasets)],
                ["Model Variants", str(n_models)],
                ["Mean Dice (Clean)", f"{mean_dice_clean:.4f}"],
                ["Mean Dice (L4)", f"{mean_dice_l4:.4f}"],
                ["Avg. Perf Drop", f"{mean_dice_clean - mean_dice_l4:.4f}"],
            ]
            
            summary_table = Table(summary_data, colWidths=[6*cm, 4*cm])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(summary_table)
        
        return story
    
    def _build_dataset_section(self) -> List:
        """Build dataset & phase settings section."""
        story = []
        
        story.append(Paragraph("Dataset & Phase Settings", self.styles['SectionTitle']))
        
        datasets = self.cfg.get("datasets", [])
        for ds in datasets:
            story.append(Paragraph(f"<b>{ds.get('name', 'Unknown')}</b>", self.styles['MyBodyText']))
            story.append(Paragraph(f"  Adapter: {ds.get('adapter', 'N/A')}", self.styles['SmallText']))
            story.append(Paragraph(f"  Root: {ds.get('root', 'N/A')}", self.styles['SmallText']))
            story.append(Spacer(1, 0.3*cm))
        
        return story
    
    def _build_noise_preset_section(self) -> List:
        """Build noise preset tables section."""
        story = []
        
        story.append(Paragraph("Noise Preset Configuration", self.styles['SectionTitle']))
        
        coupled = self.cfg.get("protocols", {}).get("coupled_presets", {})
        
        if coupled:
            story.append(Paragraph("Coupled Presets (P1)", self.styles['SubSection']))
            
            # Build table header
            levels = ["L1", "L2", "L3", "L4"]
            header = ["Noise Type"] + levels
            
            rows = [header]
            for noise_name, lv_map in coupled.items():
                row = [noise_name]
                for lv in levels:
                    if lv in lv_map:
                        params = lv_map[lv]
                        p = params.get("p", 1.0)
                        # Format params without p
                        other = {k: v for k, v in params.items() if k != "p"}
                        param_str = f"p={p:.1f}\n" + ", ".join(f"{k}={v}" for k, v in other.items())
                        row.append(param_str)
                    else:
                        row.append("-")
                rows.append(row)
            
            preset_table = Table(rows, colWidths=[3*cm] + [3.5*cm]*len(levels))
            preset_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(preset_table)
        
        # Protocol descriptions
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Protocol Descriptions", self.styles['SubSection']))
        
        protocols_desc = [
            ("P0", "Clean baseline (L0, no noise)"),
            ("P1", "Coupled Levels - both p and severity increase together"),
            ("P2a", "OFAT - sweep severity with fixed p"),
            ("P2b", "OFAT - sweep probability with fixed severity"),
            ("P3", "Grid search over p × severity combinations"),
        ]
        
        for proto, desc in protocols_desc:
            story.append(Paragraph(f"<b>{proto}:</b> {desc}", self.styles['MyBodyText']))
        
        return story
    
    def _build_quantitative_section(self, agg_df: pd.DataFrame) -> List:
        """Build quantitative results tables section."""
        story = []
        
        story.append(Paragraph("Quantitative Results", self.styles['SectionTitle']))
        
        if len(agg_df) == 0:
            story.append(Paragraph("No aggregate data available.", self.styles['MyBodyText']))
            return story
        
        # Summary table by model
        story.append(Paragraph("Results by Model (P1 Protocol)", self.styles['SubSection']))
        
        p1_data = agg_df[agg_df["protocol"] == "P1"].copy() if "protocol" in agg_df.columns else agg_df.copy()
        
        if len(p1_data) > 0:
            # Aggregate across noises and levels
            group_cols = ["dataset", "model", "weight", "mode"]
            summary = p1_data.groupby(group_cols).agg({
                "dice_mean": "mean",
                "iou_mean": "mean",
            }).reset_index()
            
            if len(summary) > 0:
                header = ["Dataset", "Model", "Weight", "Mode", "Dice (mean)", "IoU (mean)"]
                rows = [header]
                
                for _, row in summary.iterrows():
                    rows.append([
                        str(row["dataset"]),
                        str(row["model"]),
                        str(row["weight"]),
                        str(row["mode"]),
                        f"{row['dice_mean']:.4f}",
                        f"{row['iou_mean']:.4f}",
                    ])
                
                result_table = Table(rows, colWidths=[2.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
                result_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(result_table)
        
        return story
    
    def _build_per_noise_metrics_section(self, df: pd.DataFrame) -> List:
        """Build per-noise-type metrics table by level.
        
        Shows Dice, IoU for each noise type across all levels (L0-L4).
        """
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Per-Noise-Type Metrics by Level", self.styles['SectionTitle']))
        
        if len(df) == 0:
            story.append(Paragraph("No data available.", self.styles['MyBodyText']))
            return story
        
        story.append(Paragraph(
            "This section shows the mean Dice and IoU scores for each noise type at each severity level. "
            "L0 represents clean (baseline) images, while L1-L4 represent increasing noise severity.",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.3*cm))
        
        # Get unique noise types
        noise_types = df["noise"].dropna().unique().tolist()
        noise_types = [n for n in noise_types if n != "clean"]
        
        # Get unique levels  
        levels = sorted(df["level"].dropna().unique().tolist())
        
        if not noise_types or not levels:
            story.append(Paragraph("No noise data available for breakdown.", self.styles['MyBodyText']))
            return story
        
        # Process each mode separately
        modes = df["mode"].dropna().unique().tolist()
        
        for mode in modes:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(f"Mode: {mode}", self.styles['SubSection']))
            
            mode_df = df[df["mode"] == mode].copy()
            
            # Table: Noise Type | L0 Dice | L1 Dice | L2 Dice | L3 Dice | L4 Dice
            # Then a second table for IoU
            
            # DICE TABLE
            story.append(Paragraph("Dice Scores by Noise Type and Level", self.styles['SmallText']))
            header_dice = ["Noise Type"] + [f"{lv}" for lv in levels]
            rows_dice = [header_dice]
            
            for noise in noise_types:
                row_data = [noise]
                for level in levels:
                    if level == "L0":
                        # L0 is always clean from P0
                        subset = mode_df[(mode_df["protocol"] == "P0")]
                    else:
                        subset = mode_df[(mode_df["noise"] == noise) & (mode_df["level"] == level)]
                    
                    if len(subset) > 0 and "dice" in subset.columns:
                        mean_val = subset["dice"].mean()
                        row_data.append(f"{mean_val:.3f}")
                    else:
                        row_data.append("-")
                rows_dice.append(row_data)
            
            n_cols = len(header_dice)
            col_widths_dice = [3*cm] + [1.8*cm] * (n_cols - 1)
            
            dice_table = Table(rows_dice, colWidths=col_widths_dice)
            dice_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#c6f6d5')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(dice_table)
            story.append(Spacer(1, 0.3*cm))
            
            # IoU TABLE
            story.append(Paragraph("IoU Scores by Noise Type and Level", self.styles['SmallText']))
            header_iou = ["Noise Type"] + [f"{lv}" for lv in levels]
            rows_iou = [header_iou]
            
            for noise in noise_types:
                row_data = [noise]
                for level in levels:
                    if level == "L0":
                        subset = mode_df[(mode_df["protocol"] == "P0")]
                    else:
                        subset = mode_df[(mode_df["noise"] == noise) & (mode_df["level"] == level)]
                    
                    if len(subset) > 0 and "iou" in subset.columns:
                        mean_val = subset["iou"].mean()
                        row_data.append(f"{mean_val:.3f}")
                    else:
                        row_data.append("-")
                rows_iou.append(row_data)
            
            col_widths_iou = [3*cm] + [1.8*cm] * (n_cols - 1)
            
            iou_table = Table(rows_iou, colWidths=col_widths_iou)
            iou_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#bee3f8')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(iou_table)
        
        return story
    
    def _build_stability_section(self, stability_df: pd.DataFrame) -> List:
        """Build stability analysis section (basic version for backward compatibility)."""
        story = []
        
        story.append(Paragraph("Stability Analysis", self.styles['SectionTitle']))
        
        if len(stability_df) == 0:
            story.append(Paragraph("No stability data available.", self.styles['MyBodyText']))
            return story
        
        story.append(Paragraph("Performance Drop by Noise Type", self.styles['SubSection']))
        
        # Sort by perf_drop_mean descending
        sorted_df = stability_df.sort_values("perf_drop_mean", ascending=False).head(15)
        
        header = ["Noise", "Model", "Perf Drop", "Std", "N"]
        rows = [header]
        
        for _, row in sorted_df.iterrows():
            rows.append([
                str(row.get("noise", "")),
                f"{row.get('model', '')}/{row.get('weight', '')}",
                f"{row.get('perf_drop_mean', 0):.4f}",
                f"{row.get('perf_drop_std', 0):.4f}",
                str(int(row.get("n", 0))),
            ])
        
        stab_table = Table(rows, colWidths=[3*cm, 4*cm, 2.5*cm, 2.5*cm, 1.5*cm])
        stab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#fed7d7')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(stab_table)
        
        return story
    
    def _build_stability_section_extended(self, stability_df: pd.DataFrame) -> List:
        """Build extended stability analysis section with drop_Lmax, slope, AUC, CV."""
        story = []
        
        story.append(Paragraph("Stability Analysis (Extended)", self.styles['SectionTitle']))
        
        if len(stability_df) == 0:
            story.append(Paragraph("No stability data available.", self.styles['MyBodyText']))
            return story
        
        # Check for extended metrics
        extended_cols = ["drop_Lmax", "slope", "auc", "cv", "seed_std", "seed_cv"]
        has_extended = any(c in stability_df.columns for c in extended_cols)
        
        if not has_extended:
            # Fall back to basic stability section
            return self._build_stability_section(stability_df)
        
        # Extended metrics table
        story.append(Paragraph("Extended Stability Metrics", self.styles['SubSection']))
        story.append(Paragraph(
            "• <b>drop_Lmax</b>: Dice(L0) - Dice(L4) — performance drop at maximum severity<br/>"
            "• <b>slope</b>: Linear regression slope of Dice vs intensity_scalar<br/>"
            "• <b>AUC</b>: Area under curve (normalized) — higher is better<br/>"
            "• <b>CV</b>: Coefficient of variation (std/mean) across levels<br/>"
            "• <b>seed_std/cv</b>: Variability across noise seeds",
            self.styles['SmallText']
        ))
        story.append(Spacer(1, 0.3*cm))
        
        # Build table with available columns
        cols_to_show = ["noise", "model", "weight"]
        col_labels = ["Noise", "Model", "Weight"]
        
        if "drop_Lmax" in stability_df.columns:
            cols_to_show.append("drop_Lmax")
            col_labels.append("Drop_Lmax")
        if "slope" in stability_df.columns:
            cols_to_show.append("slope")
            col_labels.append("Slope")
        if "auc" in stability_df.columns:
            cols_to_show.append("auc")
            col_labels.append("AUC")
        if "cv" in stability_df.columns:
            cols_to_show.append("cv")
            col_labels.append("CV")
        
        # Sort by drop_Lmax if available
        sort_col = "drop_Lmax" if "drop_Lmax" in stability_df.columns else stability_df.columns[0]
        sorted_df = stability_df.sort_values(sort_col, ascending=False, na_position="last").head(20)
        
        rows = [col_labels]
        for _, row in sorted_df.iterrows():
            row_data = []
            for col in cols_to_show:
                val = row.get(col, "")
                if isinstance(val, float):
                    row_data.append(f"{val:.4f}" if not np.isnan(val) else "-")
                else:
                    row_data.append(str(val))
            rows.append(row_data)
        
        n_cols = len(col_labels)
        col_widths = [2.5*cm] * min(n_cols, 7)
        
        stab_table = Table(rows, colWidths=col_widths)
        stab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#fed7d7')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(stab_table)
        
        # Seed stability section (if available)
        if "seed_std" in stability_df.columns and stability_df["seed_std"].notna().any():
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("Seed Stability", self.styles['SubSection']))
            
            seed_data = stability_df[stability_df["seed_std"].notna()].copy()
            if len(seed_data) > 0:
                seed_data = seed_data.sort_values("seed_std", ascending=False).head(10)
                
                seed_rows = [["Noise", "Model", "Seed Std", "Seed CV"]]
                for _, row in seed_data.iterrows():
                    seed_rows.append([
                        str(row.get("noise", "")),
                        f"{row.get('model', '')}/{row.get('weight', '')}",
                        f"{row.get('seed_std', 0):.4f}" if not np.isnan(row.get("seed_std", np.nan)) else "-",
                        f"{row.get('seed_cv', 0):.4f}" if not np.isnan(row.get("seed_cv", np.nan)) else "-",
                    ])
                
                seed_table = Table(seed_rows, colWidths=[3*cm, 4*cm, 2.5*cm, 2.5*cm])
                seed_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(seed_table)
        
        return story
    
    def _build_uncertainty_section(self, df: pd.DataFrame) -> List:
        """Build uncertainty analysis section."""
        story = []
        
        uncertainty_cols = ["mean_confidence", "mean_entropy", "boundary_entropy", 
                          "mean_confidence_proxy", "mask_consistency_iou"]
        available = [c for c in uncertainty_cols if c in df.columns and df[c].notna().any()]
        
        if not available:
            return story
        
        story.append(PageBreak())
        story.append(Paragraph("Uncertainty Analysis", self.styles['SectionTitle']))
        
        story.append(Paragraph(
            "Uncertainty metrics help identify predictions where the model is less confident. "
            "Higher entropy and lower confidence typically correlate with worse segmentation quality.",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.3*cm))
        
        # Summary statistics
        story.append(Paragraph("Uncertainty Metrics Summary", self.styles['SubSection']))
        
        summary_rows = [["Metric", "Mean", "Std", "Min", "Max"]]
        for col in available:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary_rows.append([
                    col.replace("_", " ").title(),
                    f"{vals.mean():.4f}",
                    f"{vals.std():.4f}",
                    f"{vals.min():.4f}",
                    f"{vals.max():.4f}",
                ])
        
        if len(summary_rows) > 1:
            uncert_table = Table(summary_rows, colWidths=[4*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
            uncert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#c6f6d5')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(uncert_table)
        
        # Correlation with performance
        if "dice" in df.columns:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("Correlation with Dice Score", self.styles['SubSection']))
            
            corr_rows = [["Uncertainty Metric", "Pearson Correlation"]]
            for col in available:
                valid = df[["dice", col]].dropna()
                if len(valid) > 10:
                    corr = _safe_corrcoef(valid["dice"].values, valid[col].values)
                    corr_rows.append([col.replace("_", " ").title(), f"{corr:.4f}"])
            
            if len(corr_rows) > 1:
                corr_table = Table(corr_rows, colWidths=[5*cm, 3*cm])
                corr_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ]))
                story.append(corr_table)
        
        return story
    
    def _build_noise_gallery_section(self, gallery_paths: List[str]) -> List:
        """Build noise gallery visualization section with organized categories."""
        story = []
        
        if not gallery_paths:
            return story
        
        story.append(PageBreak())
        story.append(Paragraph("Noise Gallery & Visual Comparisons", self.styles['SectionTitle']))
        
        story.append(Paragraph(
            "This section provides visual comparisons of image quality degradation across different noise types and severity levels. "
            "These visualizations help understand how each noise type affects the segmentation performance.",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # Filter to only valid image files (not PDFs)
        valid_gallery_paths = [p for p in gallery_paths if self._is_valid_image(p)]
        
        if not valid_gallery_paths:
            story.append(Paragraph("No valid gallery images found.", self.styles['MyBodyText']))
            return story
        
        # Categorize images by type
        comprehensive = [p for p in valid_gallery_paths if "comprehensive" in p.lower()]
        noise_type_galleries = [p for p in valid_gallery_paths if "noise_gallery_" in p.lower()]
        noise_viz = [p for p in valid_gallery_paths if "noise_visualization" in p.lower() or "noise_viz" in p.lower()]
        noise_effect = [p for p in valid_gallery_paths if "noise_effect" in p.lower()]
        others = [p for p in valid_gallery_paths if p not in comprehensive + noise_type_galleries + noise_viz + noise_effect]
        
        def add_gallery_group(paths: List[str], title: str, description: str, max_images: int = 6):
            """Add a group of gallery images with title and description."""
            if not paths:
                return
            
            story.append(Paragraph(title, self.styles['SubSection']))
            story.append(Paragraph(description, self.styles['SmallText']))
            story.append(Spacer(1, 0.3*cm))
            
            for i, img_path in enumerate(paths[:max_images]):
                try:
                    # Determine appropriate size based on image type
                    if "comprehensive" in img_path.lower():
                        img = RLImage(img_path, width=17*cm, height=20*cm)
                    else:
                        img = RLImage(img_path, width=16*cm, height=10*cm)
                    story.append(img)
                    story.append(Spacer(1, 0.3*cm))
                    
                    # Page break after each large image
                    if i < len(paths) - 1:
                        story.append(PageBreak())
                except Exception:
                    continue
        
        # Add comprehensive comparisons first (most detailed)
        add_gallery_group(
            comprehensive,
            "Comprehensive Noise Comparisons",
            "Each grid shows all noise types (rows) across all severity levels (columns) for a single sample. "
            "Each cell shows the noisy image and prediction overlay with Dice score.",
            max_images=4
        )
        
        # Add noise type galleries
        add_gallery_group(
            noise_type_galleries,
            "Per-Noise-Type Galleries",
            "Each image shows multiple samples (rows) for a single noise type across severity levels (columns). "
            "Dice scores are color-coded: green (>0.8), orange (0.5-0.8), red (<0.5).",
            max_images=8
        )
        
        # Add noise visualization
        add_gallery_group(
            noise_viz,
            "Noise Visualization Samples",
            "Visual examples of how different noise types affect image quality at various severity levels.",
            max_images=6
        )
        
        # Add noise effect comparisons
        add_gallery_group(
            noise_effect,
            "Noise Effect Comparisons",
            "Side-by-side comparison showing the effect of each noise type on segmentation.",
            max_images=4
        )
        
        # Add other images
        if others:
            add_gallery_group(
                others,
                "Additional Visualizations",
                "Other noise-related visualizations.",
                max_images=4
            )
        
        return story
    
    def _build_global_plots_section(self, plot_paths: List[str]) -> List:
        """Build global comparative plots section."""
        story = []
        
        if not plot_paths:
            return story
        
        story.append(PageBreak())
        story.append(Paragraph("Global Comparative Analysis", self.styles['SectionTitle']))
        
        story.append(Paragraph(
            "These plots provide a global view of model sensitivity across all noise types, "
            "including sensitivity heatmaps, ranking charts, and PSNR/performance correlations.",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # Filter to only valid image files
        valid_plot_paths = [p for p in plot_paths if self._is_valid_image(p)]
        
        # Categorize plots by type
        heatmaps = [p for p in valid_plot_paths if "heatmap" in p.lower()]
        rankings = [p for p in valid_plot_paths if "ranking" in p.lower()]
        curves = [p for p in valid_plot_paths if "curve" in p.lower()]
        others = [p for p in valid_plot_paths if p not in heatmaps + rankings + curves]
        
        def add_plot_group(paths: List[str], title: str, max_plots: int = 4):
            if not paths:
                return
            story.append(Paragraph(title, self.styles['SubSection']))
            for i, img_path in enumerate(paths[:max_plots]):
                try:
                    img = RLImage(img_path, width=14*cm, height=8*cm)
                    story.append(img)
                    story.append(Spacer(1, 0.3*cm))
                    if (i + 1) % 2 == 0 and i < len(paths) - 1:
                        story.append(PageBreak())
                except Exception:
                    continue
        
        add_plot_group(heatmaps, "Sensitivity Heatmaps")
        add_plot_group(rankings, "Impact Rankings")
        add_plot_group(curves, "Sensitivity Curves")
        add_plot_group(others, "Additional Plots")
        
        return story
    
    def _build_mode_comparison_section(self, figure_paths: List[str]) -> List:
        """Build mode comparison (automatic vs prompt_bbox) plots section with organized categories."""
        story = []
        
        # Filter for mode comparison plots
        mode_plots = [p for p in figure_paths if self._is_valid_image(p) and 
                      any(kw in p.lower() for kw in ["mode_comparison", "by_mode", "automatic_vs", "prompt_bbox"])]
        
        if not mode_plots:
            return story
        
        story.append(PageBreak())
        story.append(Paragraph("Mode Comparison Analysis", self.styles['SectionTitle']))
        
        story.append(Paragraph(
            "Comparison of segmentation performance between automatic mode (no prompts) "
            "and prompt-guided mode (bounding box prompts). These plots help understand "
            "how prompt guidance affects model robustness to noise.",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # Categorize mode comparison plots
        summary_plots = [p for p in mode_plots if "summary" in p.lower()]
        grouped_plots = [p for p in mode_plots if "grouped" in p.lower() and "summary" not in p.lower()]
        per_noise_plots = [p for p in mode_plots if p not in summary_plots + grouped_plots]
        
        def add_mode_plot_group(paths: List[str], title: str, description: str, max_plots: int = 6):
            """Add a group of mode comparison plots."""
            if not paths:
                return
            
            story.append(Paragraph(title, self.styles['SubSection']))
            story.append(Paragraph(description, self.styles['SmallText']))
            story.append(Spacer(1, 0.3*cm))
            
            for i, img_path in enumerate(paths[:max_plots]):
                try:
                    img = RLImage(img_path, width=15*cm, height=9*cm)
                    story.append(img)
                    story.append(Spacer(1, 0.3*cm))
                    
                    if (i + 1) % 2 == 0 and i < len(paths) - 1:
                        story.append(PageBreak())
                except Exception:
                    continue
        
        # 1. Summary plots first (overall comparison)
        add_mode_plot_group(
            summary_plots,
            "Overall Mode Comparison Summary",
            "Mean performance (±std) across all noise types for each level, comparing automatic vs prompt_bbox modes.",
            max_plots=4
        )
        
        # 2. Grouped plots (noise types at each level)
        add_mode_plot_group(
            grouped_plots,
            "Mode Comparison by Level (All Noise Types)",
            "Grouped bar charts showing performance for each noise type at specific levels (L0-L4), comparing modes.",
            max_plots=8
        )
        
        # 3. Per-noise plots
        add_mode_plot_group(
            per_noise_plots,
            "Mode Comparison per Noise Type",
            "Detailed comparison curves for each individual noise type across all levels.",
            max_plots=6
        )
        
        return story
    
    def _build_plots_section(self, figure_paths: List[str]) -> List:
        """Build sensitivity plots section."""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Sensitivity Plots", self.styles['SectionTitle']))
        
        if not figure_paths:
            story.append(Paragraph("No plots generated.", self.styles['MyBodyText']))
            return story
        
        # Filter to only include image files (not PDFs)
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        valid_figure_paths = [
            p for p in figure_paths 
            if Path(p).suffix.lower() in image_extensions and Path(p).exists()
        ]
        
        if not valid_figure_paths:
            story.append(Paragraph("No image plots found (only PDF plots available).", self.styles['MyBodyText']))
            return story
        
        # Add up to 12 plots
        for i, fig_path in enumerate(valid_figure_paths[:12]):
            try:
                img = RLImage(fig_path, width=14*cm, height=8*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))
                
                # Add page break every 2 images
                if (i + 1) % 2 == 0 and i < len(valid_figure_paths) - 1:
                    story.append(PageBreak())
            except Exception:
                continue
        
        return story
    
    def _build_failure_section(self, failure_paths: List[str]) -> List:
        """Build failure cases section."""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Failure Case Analysis", self.styles['SectionTitle']))
        
        story.append(Paragraph(
            "The following samples show the largest performance degradation from clean (L0) to severe noise (L4).",
            self.styles['MyBodyText']
        ))
        story.append(Spacer(1, 0.5*cm))
        
        if not failure_paths:
            story.append(Paragraph("No failure cases to display.", self.styles['MyBodyText']))
            return story
        
        # Filter to only valid image files
        valid_failure_paths = [p for p in failure_paths if self._is_valid_image(p)]
        
        for i, fail_path in enumerate(valid_failure_paths[:8]):
            try:
                img = RLImage(fail_path, width=16*cm, height=5*cm)
                story.append(img)
                story.append(Spacer(1, 0.3*cm))
                
                if (i + 1) % 3 == 0:
                    story.append(PageBreak())
            except Exception:
                continue
        
        return story
    
    def _build_appendix(self) -> List:
        """Build appendix section."""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Appendix", self.styles['SectionTitle']))
        
        story.append(Paragraph("Output Files", self.styles['SubSection']))
        
        files = [
            ("results.csv", "Per-sample results with all metrics"),
            ("aggregate.csv", "Aggregated results by group"),
            ("stability.csv", "Stability metrics summary"),
            ("summary.json", "Experiment summary"),
            ("figures/", "All generated plots"),
            ("pred_masks/", "Saved prediction masks"),
        ]
        
        for fname, desc in files:
            story.append(Paragraph(f"• <b>{fname}</b>: {desc}", self.styles['MyBodyText']))
        
        story.append(Spacer(1, 1*cm))
        # story.append(Paragraph(
        #     #f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        #     self.styles['SmallText']
        # ))
        
        return story


def build_report_pdf(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    cfg: dict,
    exp_dir: Path,
    figure_paths: List[str],
    failure_paths: List[str],
    noise_gallery_paths: List[str] = None,
    global_plot_paths: List[str] = None
) -> Path:
    """
    Build the complete PDF report.
    
    Args:
        df: Per-sample results
        agg_df: Aggregated results
        stability_df: Stability metrics
        cfg: Config dictionary
        exp_dir: Experiment output directory
        figure_paths: List of figure image paths
        failure_paths: List of failure case image paths
        noise_gallery_paths: List of noise gallery image paths
        global_plot_paths: List of global comparative plot paths
        
    Returns:
        Path to generated PDF
    """
    exp_dir = Path(exp_dir)
    out_path = exp_dir / "report.pdf"
    
    builder = PDFReportBuilder(cfg, exp_dir)
    builder.build_report(
        df=df,
        agg_df=agg_df,
        stability_df=stability_df,
        figure_paths=figure_paths,
        failure_paths=failure_paths,
        out_path=out_path,
        noise_gallery_paths=noise_gallery_paths or [],
        global_plot_paths=global_plot_paths or []
    )
    
    return out_path
