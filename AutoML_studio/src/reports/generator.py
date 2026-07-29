import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    KeepTogether
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReportGenerator:

    def __init__(self, output_directory: str = "generated_reports"):
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        logger.info(f"ReportGenerator initialized. Output folder: '{self.output_directory}'")

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#2B6CB0'),
            spaceBefore=14,
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            name='BodyDark',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#2D3748')
        ))
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=colors.white,
            alignment=1  # Center aligned
        ))
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#1A202C')
        ))

    def _build_table_from_dict(self, data_dict: Dict[str, Any], col_widths: List[float]) -> Table:
        table_data = [[
            Paragraph(f"<b>{k}</b>", self.styles['TableCell']),
            Paragraph(str(v), self.styles['TableCell'])
        ] for k, v in data_dict.items()]

        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        return t

    def _build_leaderboard_table(self, df: pd.DataFrame) -> Table:
        display_cols = df.columns[:6]
        header_row = [Paragraph(col, self.styles['TableHeader']) for col in display_cols]
        table_data = [header_row]

        for _, row in df.iterrows():
            data_row = []
            for col in display_cols:
                val = row[col]
                if isinstance(val, float):
                    formatted_val = f"{val:.4f}"
                else:
                    formatted_val = str(val)
                data_row.append(Paragraph(formatted_val, self.styles['TableCell']))
            table_data.append(data_row)

        col_width = (6.5 * inch) / len(display_cols)
        t = Table(table_data, colWidths=[col_width] * len(display_cols))
        
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        return t

    def generate_report(
        self,
        project_name: str,
        task_type: str,
        dataset_summary: Dict[str, Any],
        cleaning_summary: Dict[str, Any],
        leaderboard_df: pd.DataFrame,
        recommendation_text: str,
        shap_plot_path: Optional[str] = None
    ) -> str:
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"AutoML_Report_{project_name.replace(' ', '_')}_{timestamp_str}.pdf"
            file_path = os.path.join(self.output_directory, filename)

            doc = SimpleDocTemplate(
                file_path,
                pagesize=letter,
                rightMargin=0.75 * inch,
                leftMargin=0.75 * inch,
                topMargin=0.75 * inch,
                bottomMargin=0.75 * inch
            )

            story = []

            # 1. Document Header & Metadata
            story.append(Paragraph("AutoML Studio: Executive Analysis Report", self.styles['ReportTitle']))
            story.append(Paragraph(f"<b>Project:</b> {project_name} | <b>Task:</b> {task_type.upper()} | <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['BodyDark']))
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A365D'), spaceBefore=1, spaceAfter=15))

            # 2. Dataset Overview Section
            story.append(Paragraph("1. Dataset Architecture & Health Overview", self.styles['SectionHeader']))
            story.append(self._build_table_from_dict(dataset_summary, col_widths=[2.5 * inch, 4.0 * inch]))
            story.append(Spacer(1, 12))

            # 3. Data Cleaning & Preprocessing Audit
            story.append(Paragraph("2. Automated Data Preprocessing Audit", self.styles['SectionHeader']))
            story.append(self._build_table_from_dict(cleaning_summary, col_widths=[2.5 * inch, 4.0 * inch]))
            story.append(Spacer(1, 15))

            # 4. Model Evaluation Leaderboard
            story.append(Paragraph("3. Model Benchmarking Leaderboard", self.styles['SectionHeader']))
            story.append(Paragraph("Models evaluated and ranked by composite performance across accuracy, stability, and speed:", self.styles['BodyDark']))
            story.append(Spacer(1, 8))
            story.append(self._build_leaderboard_table(leaderboard_df))
            story.append(Spacer(1, 15))

            # 5. Executive Model Recommendation
            rec_elements = [
                Paragraph("4. AI Recommendation & Justification", self.styles['SectionHeader']),
                Paragraph(recommendation_text.replace("\n", "<br/>"), self.styles['BodyDark']),
                Spacer(1, 15)
            ]
            story.append(KeepTogether(rec_elements))

            # 6. Explainable AI (SHAP Visualization)
            if shap_plot_path and os.path.exists(shap_plot_path):
                shap_elements = [
                    Paragraph("5. Global Feature Importance (SHAP Analysis)", self.styles['SectionHeader']),
                    Paragraph("The visualization below details the primary behavioral drivers influencing the model's predictions:", self.styles['BodyDark']),
                    Spacer(1, 8),
                    Image(shap_plot_path, width=6.0 * inch, height=3.5 * inch),
                    Spacer(1, 15)
                ]
                story.append(KeepTogether(shap_elements))
            elif shap_plot_path:
                logger.warning(f"SHAP plot path provided but file does not exist: {shap_plot_path}")

            # Build document
            logger.info("Compiling elements and rendering PDF...")
            doc.build(story)
            logger.info(f"PDF Report successfully generated at: {file_path}")
            
            return os.path.abspath(file_path)

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            raise e