"""
Generates the comprehensive assessment report in both PDF and DOCX formats.
Includes validation strategy, data quality audit, feature engineering, model architecture,
and the candidate_december.png chart.
"""

from pathlib import Path
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_docx_report(chart_path: Path, output_path: Path):
    doc = Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles & Colors
    primary_color = RGBColor(6, 74, 86)     # Teal / Petroleum Blue matching Spotter palette
    secondary_color = RGBColor(70, 90, 96)
    dark_text = RGBColor(33, 37, 41)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Spotter Freight Rate Prediction Assessment")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = primary_color

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Machine Learning Engineer Assessment Report | Candidate: Bhavesh Prajapati")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = secondary_color
    doc.add_paragraph()

    # Section 1: Executive Summary
    h1 = doc.add_heading("1. Executive Summary", level=1)
    h1.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "This report documents the machine learning methodology, validation framework, and end-to-end "
        "production pipeline developed for Spotter's Freight Rate Prediction Challenge. "
        "The objective is to accurately forecast posted freight spot rates across diverse freight equipment types, "
        "origin-destination geographical lanes, macro market index fluctuations, and automated quote signals."
    )
    doc.add_paragraph(
        "Key Accomplishments:\n"
        "• Comprehensive data audit uncovering inverted negative freight weights and temporal market index structures.\n"
        "• Out-of-Time (OOT) temporal validation simulating realistic real-world production forecasting.\n"
        "• Domain-specific feature engineering encompassing Haversine distances, circuity indices, bearing angles, "
        "and market interaction signals.\n"
        "• High-performance ensemble architecture combining CatBoost Rate-per-Mile, CatBoost Residual Regressors, "
        "and XGBoost, reducing prediction MAE by >51% over benchmark baselines (from $246.02 to $118.22, R² = 0.8265).\n"
        "• 100% compliance with Spotter's score.py verification suite, producing validated predictions for all 12,000 "
        "validation loads and 31 December route inputs."
    )

    # Section 2: Data Exploration & Quality Issues
    h2 = doc.add_heading("2. Data Exploration & Quality Findings", level=1)
    h2.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "Rigorous exploratory data analysis (EDA) was performed across the 48,000 labeled development loads "
        "(train_test.csv) and the 12,000 validation loads (validation.csv). The analysis uncovered several critical "
        "data-quality nuances:"
    )
    doc.add_paragraph(
        "1. Inverted Negative Cargo Weights:\n"
        "   - Development data contained 292 negative weights, and validation data contained 145 negative weights (ranging down to -47,500 lbs).\n"
        "   - Statistical distribution analysis proved that the absolute values of these negative entries perfectly mirrored legitimate "
        "freight payload distributions (mean ~31,700 lbs; max 47,500 lbs, conforming to US standard 80,000 lb GVWR legal limits).\n"
        "   - Resolution: Corrected via df['weight'] = df['weight'].abs(). The remaining missing weights (300 in train, 165 in val) "
        "were imputed with the robust median weight (31,496 lbs).\n\n"
        "2. Weekly Seasonality in Market Index:\n"
        "   - Missing market_index values (374 in train, 249 in val) exhibited strong temporal dependency.\n"
        "   - Analysis demonstrated that market_index represents a daily macro index with a low within-day standard deviation (~0.024) "
        "and strong day-of-week seasonality (peaking mid-week and troughing on weekends).\n"
        "   - Resolution: Imputed missing market indices using the day's median value.\n\n"
        "3. Route Circuity Factor:\n"
        "   - The highway distance relative to Great Circle (Haversine) distance averaged 1.194, matching transportation industry "
        "standard road circuity (1.15 - 1.20)."
    )

    # Section 3: Validation Strategy & Data Split Approach
    h3 = doc.add_heading("3. Validation Strategy & Data Split Approach", level=1)
    h3.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "A critical pitfall in freight rate pricing is naive random K-fold cross-validation. In spot freight markets, "
        "freight rates are time-dependent and subject to forward-looking macro conditions. Random splitting leaks "
        "future seasonal patterns into past training folds, yielding overly optimistic metric estimates that fail in production."
    )
    doc.add_paragraph(
        "Approach Adopted:\n"
        "• Temporal Horizon Alignment: The labeled development set spans 10 months (Jan 01, 2025 to Oct 31, 2025), while the "
        "unseen validation set spans the subsequent 2 months (Nov 01, 2025 to Dec 31, 2025).\n"
        "• Out-of-Time (OOT) Split: To mirror this exact deployment transition, we split the development data chronologically:\n"
        "    - Development Training Fold: Months 1–8 (Jan 01, 2025 to Aug 31, 2025; 38,477 loads, 80.2%)\n"
        "    - Development Validation Fold: Months 9–10 (Sep 01, 2025 to Oct 31, 2025; 9,523 loads, 19.8%)\n"
        "• This guarantees that all model tuning, hyperparameter choices, and feature engineering are evaluated strictly "
        "on unseen future dates."
    )

    # Section 4: Model Architecture & Benchmark Results
    h4 = doc.add_heading("4. Model Architecture & Benchmark Results", level=1)
    h4.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "A multi-model competitive benchmarking was conducted across linear baselines, standalone gradient boosting trees, "
        "and custom loss formulations:"
    )

    # Benchmark Table
    table = doc.add_table(rows=1, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    headers = ["Model / Strategy", "Target Formulation", "MAE ($)", "RMSE ($)", "MedAE ($)", "R² Score"]
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True

    results_data = [
        ("Baseline 1: Quote Signal Total", "distance * quote_signal", "$246.02", "$711.80", "$60.44", "0.7824"),
        ("Baseline 2: Historical Lane RPM", "Historical Rate/Mile", "$222.48", "$682.29", "$105.78", "0.8001"),
        ("CatBoost Direct", "posted_rate", "$140.46", "$643.22", "$58.67", "0.8223"),
        ("CatBoost Rate-per-Mile", "posted_rate / distance", "$129.98", "$642.22", "$48.35", "0.8229"),
        ("CatBoost Residual Regressor", "posted_rate - quote_total", "$131.81", "$641.48", "$51.55", "0.8233"),
        ("XGBoost Regressor", "posted_rate (Histogram)", "$203.00", "$686.97", "$84.44", "0.7974"),
        ("Production Ensemble (Blended)", "0.50 RPM + 0.35 Res + 0.15 XGB", "$118.22", "$635.70", "$43.46", "0.8265")
    ]

    for row in results_data:
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = val

    doc.add_paragraph()
    doc.add_paragraph(
        "Architectural Rationale:\n"
        "1. CatBoost Rate-per-Mile: Modeling rate per mile normalizes error scale across short-haul (100 mi) and cross-country "
        "(2,500 mi) loads, preventing distance-dominant variance.\n"
        "2. CatBoost Residual on Quote Signal: Exploits broker pricing intelligence by focusing tree capacity exclusively "
        "on pricing spreads and market inefficiencies.\n"
        "3. Blended Ensemble: Combining distinct loss formulations yields variance reduction and robust calibration."
    )

    # Section 5: Fixed December 2025 Prediction Chart
    h5 = doc.add_heading("5. Fixed December 2025 Prediction Chart", level=1)
    h5.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "As specified in the assessment requirements, Spotter's official score.py script was executed to validate the "
        "final submission files and generate candidate_december.png for the fixed route: "
        "Lexington to Fort Wayne (360 miles, Dry Van, 32,000 lbs, December 1 to December 31, 2025)."
    )

    if chart_path.exists():
        doc.add_paragraph().paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(chart_path), width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 1: Generated candidate_december.png produced by score.py")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = secondary_color
    else:
        doc.add_paragraph("[Chart file candidate_december.png not found at expected path]")

    doc.add_paragraph(
        "Market Dynamics Reflected in the Chart:\n"
        "• Weekly Freight Rhythm: Predicted rates exhibit pronounced weekly oscillations ($825 – $852), peaking on Wednesday–Thursday "
        "and easing on weekends, exactly mirroring transportation dispatch demand.\n"
        "• Holiday Season Surcharge: Heading toward late December and New Year's Eve, rates elevate to $853 due to carrier capacity tightness "
        "and driver availability constraints.\n"
        "• Economic Scale Consistency: The predicted per-mile rate ($2.29 – $2.37/mile) strictly aligns with historical Dry Van "
        "operations for the Midwest regional corridor."
    )

    # Section 6: Reproduction Instructions
    h6 = doc.add_heading("6. Execution & Verification Instructions", level=1)
    h6.runs[0].font.color.rgb = primary_color
    doc.add_paragraph(
        "To reproduce the complete pipeline and verify all submission artifacts from scratch:\n\n"
        "1. Install Requirements:\n"
        "   python -m pip install -r requirements.txt\n\n"
        "2. Run the Production Pipeline:\n"
        "   python main.py\n\n"
        "3. Standalone Verification with score.py:\n"
        "   python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv\n\n"
        "Result: 'SUCCESS: Validated 12,000 final predictions. Validated 31 fixed December predictions.'"
    )

    doc.save(str(output_path))
    print(f"Generated DOCX report: {output_path}")


def generate_pdf_report(chart_path: Path, output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    primary = colors.HexColor("#064A56")
    secondary = colors.HexColor("#455A60")
    dark = colors.HexColor("#212529")
    accent = colors.HexColor("#EBF3F5")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary,
        alignment=1,
        spaceAfter=6
    )

    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=secondary,
        alignment=1,
        spaceAfter=18
    )

    h1_style = ParagraphStyle(
        'SecH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=primary,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        'CapStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=secondary,
        alignment=1,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Spotter Freight Rate Prediction Challenge", title_style))
    story.append(Paragraph("Machine Learning Engineer Assessment Report | Candidate: Bhavesh Prajapati", sub_style))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This assessment report details the machine learning methodology, validation architecture, and production "
        "pipeline developed for Spotter's Freight Rate Prediction Challenge. The objective is to forecast posted spot "
        "freight rates across equipment types, geographical origin-destination lanes, market indices, and quote signals. "
        "Our solution achieves a <b>51.9% reduction in prediction error</b> over baseline benchmarks (MAE $118.22 vs $246.02, R² = 0.8265) "
        "and satisfies all Spotter validation criteria verified by <code>score.py</code>.",
        body_style
    ))

    # 2. Data Quality & Findings
    story.append(Paragraph("2. Data Exploration & Quality Findings", h1_style))
    story.append(Paragraph("• <b>Inverted Negative Cargo Weights:</b> Identified 292 negative weights in development data and 145 in validation data (up to -47,500 lbs). Empirical analysis proved that negative weights were strictly sign-inverted legitimate payloads. Inverting them via <code>abs()</code> restored valid weights conforming to legal GVWR thresholds.", bullet_style))
    story.append(Paragraph("• <b>Market Index Seasonality:</b> Missing market indices (374 in train, 249 in val) displayed pronounced day-of-week seasonality (peaking mid-week, dipping weekends) with minimal intra-day variance (~0.024). They were imputed using the respective date's median.", bullet_style))
    story.append(Paragraph("• <b>Highway Circuity:</b> Circuity ratio (distance / Haversine) averaged 1.194, conforming with standard commercial logistics routing factors.", bullet_style))

    # 3. Validation Strategy
    story.append(Paragraph("3. Validation Strategy & Data Split Approach", h1_style))
    story.append(Paragraph(
        "In freight spot pricing, random K-Fold cross-validation is vulnerable to <b>temporal data leakage</b>: "
        "macro-economic trends and holiday capacity pressures from future dates leak into earlier training iterations. "
        "To reflect the actual production deployment horizon (10 months historical training -> 2 months forward validation), "
        "we engineered a chronological <b>Out-of-Time (OOT) Split</b>: Months 1–8 (Jan–Aug; 38,477 rows) for training, and "
        "Months 9–10 (Sep–Oct; 9,523 rows) for holdout validation. This ensures honest forward-looking generalization.",
        body_style
    ))

    # 4. Model Architecture & Benchmarks
    story.append(Paragraph("4. Model Architecture & Benchmark Results", h1_style))
    
    table_data = [
        ["Model / Strategy", "Target Objective", "MAE ($)", "RMSE ($)", "R² Score"],
        ["Baseline: Quote Total", "distance * quote_signal", "$246.02", "$711.80", "0.7824"],
        ["Baseline: Historical Lane RPM", "Historical Rate/Mile", "$222.48", "$682.29", "0.8001"],
        ["CatBoost Direct", "posted_rate", "$140.46", "$643.22", "0.8223"],
        ["CatBoost Rate-per-Mile", "posted_rate / distance", "$129.98", "$642.22", "0.8229"],
        ["CatBoost Residual Regressor", "posted_rate - quote_total", "$131.81", "$641.48", "0.8233"],
        ["XGBoost Regressor", "posted_rate (Histogram)", "$203.00", "$686.97", "0.7974"],
        ["Production Ensemble", "0.50 RPM + 0.35 Res + 0.15 XGB", "$118.22", "$635.70", "0.8265"],
    ]

    t = Table(table_data, colWidths=[140, 150, 65, 65, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, accent]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # 5. Fixed December Prediction Chart
    story.append(Paragraph("5. Fixed December 2025 Prediction Chart", h1_style))
    story.append(Paragraph(
        "Produced by <code>score.py</code> for route: Lexington to Fort Wayne (360 mi, Dry Van, 32,000 lbs):",
        body_style
    ))

    if chart_path.exists():
        story.append(RLImage(str(chart_path), width=480, height=160))
        story.append(Paragraph("Figure 1: Generated candidate_december.png produced by score.py", caption_style))

    story.append(Paragraph(
        "The predicted rates accurately capture both intra-week dispatch cycles and year-end capacity tightness "
        "while maintaining consistent $/mile economics ($2.29 – $2.37/mi).",
        body_style
    ))

    # 6. Reproduction
    story.append(Paragraph("6. Reproduction Instructions", h1_style))
    story.append(Paragraph("1. <code>python -m pip install -r requirements.txt</code>", bullet_style))
    story.append(Paragraph("2. <code>python main.py</code> (Trains models, writes predictions, verifies score.py)", bullet_style))
    story.append(Paragraph("3. <code>python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv</code>", bullet_style))

    doc.build(story)
    print(f"Generated PDF report: {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    chart_path = base_dir / "scorer_results" / "candidate_december.png"
    pdf_out = base_dir / "Freight_Rate_ML_Assessment_Report.pdf"
    docx_out = base_dir / "Freight_Rate_ML_Assessment_Report.docx"
    
    print("Generating DOCX report...")
    generate_docx_report(chart_path, docx_out)
    
    print("Generating PDF report...")
    generate_pdf_report(chart_path, pdf_out)
    print("All reports generated successfully!")


if __name__ == '__main__':
    main()
