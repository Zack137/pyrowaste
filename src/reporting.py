from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Mapping

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import DEFAULT_WEIGHTS, DISPLAY_NAMES


GREEN = colors.HexColor("#176B5B")
DARK = colors.HexColor("#173D38")
ORANGE = colors.HexColor("#D18A3A")
PALE = colors.HexColor("#EAF3EF")
LIGHT = colors.HexColor("#F6F8F7")
RED = colors.HexColor("#B84A3A")


def _number(value: object, decimals: int = 3) -> str:
    try:
        return f"{float(value):,.{decimals}f}".replace(",", " ")
    except (TypeError, ValueError):
        return str(value)


def _safe_change(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return (after - before) / abs(before) * 100.0


def _improvement_chart(before: Mapping[str, object], after: Mapping[str, object]) -> Drawing:
    metrics = [
        ("Liquide", "Concentration_liquide_chimique_kg", True),
        ("Toxicité", "Toxicity_Index_pct", True),
        ("Coût liquide", "Cout_liquide_USD", True),
        ("Score", "Score_final", False),
    ]
    improvements = []
    for label, key, minimize in metrics:
        change = _safe_change(float(before[key]), float(after[key]))
        improvements.append((label, -change if minimize else change))

    width, height = 480, 145
    drawing = Drawing(width, height)
    drawing.add(String(0, 130, "Amélioration relative après optimisation", fontSize=11, fillColor=DARK))
    max_value = max(100.0, max(abs(value) for _, value in improvements))
    for index, (label, value) in enumerate(improvements):
        y = 95 - index * 27
        drawing.add(String(0, y + 4, label, fontSize=8.5, fillColor=DARK))
        bar_width = max(0.0, min(300.0, 300.0 * value / max_value))
        drawing.add(Rect(100, y, 300, 13, fillColor=colors.HexColor("#E4EAE7"), strokeColor=None))
        drawing.add(Rect(100, y, bar_width, 13, fillColor=GREEN if value >= 0 else RED, strokeColor=None))
        drawing.add(String(410, y + 3, f"{value:+.1f} %", fontSize=8.5, fillColor=DARK))
    return drawing


def build_optimization_pdf(
    *,
    strategy: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    trials: int,
    weights: Mapping[str, float] | None = None,
    selected_models_by_target: Mapping[str, str] | None = None,
) -> bytes:
    """Génère un rapport PDF d'une prédiction suivie d'une optimisation."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title="Rapport PyroWaste",
        author="PyroWaste",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=GREEN,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=DARK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#4D625D"),
        )
    )

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E3DF"))
        canvas.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
        canvas.setFillColor(colors.HexColor("#60746F"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(16 * mm, 9 * mm, "PyroWaste — rapport de scénario")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph("PyroWaste", styles["ReportTitle"]),
        Paragraph(
            "Rapport de prédiction et d’optimisation multicritère",
            styles["ReportBody"],
        ),
        Spacer(1, 3 * mm),
    ]

    metadata = [
        ["Date de génération", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Stratégie de modèle", strategy],
        ["Essais Optuna — recette", str(int(trials))],
    ]
    metadata_table = Table(metadata, colWidths=[47 * mm, 120 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE),
                ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFDBD6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metadata_table, Spacer(1, 5 * mm)])
    if selected_models_by_target:
        model_rows = [["SORTIE", "MODÈLE RETENU"]]
        for target, model in selected_models_by_target.items():
            model_rows.append([DISPLAY_NAMES.get(target, target), model])
        model_table = Table(
            model_rows,
            colWidths=[78 * mm, 82 * mm],
            repeatRows=1,
        )
        model_table.setStyle(_table_style())
        story.extend(
            [
                Paragraph(
                    "Sélection automatique par sortie",
                    styles["ReportSection"],
                ),
                model_table,
                Spacer(1, 3 * mm),
            ]
        )

    score_before = float(before["Score_final"])
    score_after = float(after["Score_final"])
    toxicity_before = float(before["Toxicity_Index_pct"])
    toxicity_after = float(after["Toxicity_Index_pct"])
    liquid_before = float(before["Concentration_liquide_chimique_kg"])
    liquid_after = float(after["Concentration_liquide_chimique_kg"])
    kpis = [
        ["INDICATEUR", "AVANT", "APRÈS", "ÉVOLUTION"],
        ["Score final", _number(score_before, 1), _number(score_after, 1), f"{score_after-score_before:+.1f} pts"],
        ["Liquide chimique", f"{_number(liquid_before)} kg", f"{_number(liquid_after)} kg", f"{_safe_change(liquid_before, liquid_after):+.1f} %"],
        ["Toxicity Index", f"{_number(toxicity_before)} %", f"{_number(toxicity_after)} %", f"{_safe_change(toxicity_before, toxicity_after):+.1f} %"],
        ["Coût liquide", f"{_number(before['Cout_liquide_USD'], 2)} $", f"{_number(after['Cout_liquide_USD'], 2)} $", f"{_safe_change(float(before['Cout_liquide_USD']), float(after['Cout_liquide_USD'])):+.1f} %"],
    ]
    kpi_table = Table(kpis, colWidths=[50 * mm, 36 * mm, 36 * mm, 35 * mm], repeatRows=1)
    kpi_table.setStyle(_table_style())
    story.extend(
        [
            Paragraph("Synthèse", styles["ReportSection"]),
            kpi_table,
            Spacer(1, 3 * mm),
            _improvement_chart(before, after),
        ]
    )

    input_columns = [
        ("Liquide chimique (kg)", "Concentration_liquide_chimique_kg"),
        ("Température (°C)", "Temperature_C"),
        ("Temps réaction (min)", "Temps_reaction_min"),
        ("Heating rate (°C/min)", "Heating_rate_C_min"),
        ("Taille particules (mm)", "Taille_particules_mm"),
    ]
    input_rows = [["CONDITION", "AVANT", "APRÈS", "ÉCART"]]
    for label, key in input_columns:
        old, new = float(before[key]), float(after[key])
        input_rows.append([label, _number(old), _number(new), f"{new-old:+.3f}"])
    input_table = Table(input_rows, colWidths=[59 * mm, 35 * mm, 35 * mm, 31 * mm], repeatRows=1)
    input_table.setStyle(_table_style())
    story.extend(
        [
            Paragraph("Conditions opératoires", styles["ReportSection"]),
            input_table,
        ]
    )

    output_columns = [
        ("Solid Yield (%)", "Solid_Yield_pct"),
        ("Liquid Yield (%)", "Liquid_Yield_pct"),
        ("Gas Yield (%)", "Gas_Yield_pct"),
        ("HBr (%)", "HBr_pct"),
        ("Br2 (%)", "Br2_pct"),
        ("HCl (%)", "HCl_pct"),
        ("Cl2 (%)", "Cl2_pct"),
        ("HF (%)", "HF_pct"),
        ("Toxicity Index (%)", "Toxicity_Index_pct"),
        ("Coût liquide ($)", "Cout_liquide_USD"),
        ("Coût énergie ($)", "Cout_energie_USD"),
        ("Score final", "Score_final"),
    ]
    output_rows = [["SORTIE", "AVANT", "APRÈS", "ÉCART"]]
    for label, key in output_columns:
        old, new = float(before[key]), float(after[key])
        output_rows.append([label, _number(old), _number(new), f"{new-old:+.3f}"])
    output_table = Table(output_rows, colWidths=[59 * mm, 35 * mm, 35 * mm, 31 * mm], repeatRows=1)
    output_table.setStyle(_table_style())
    story.extend(
        [
            Paragraph("Prédictions et résultats", styles["ReportSection"]),
            output_table,
            PageBreak(),
            Paragraph("Fonction objectif", styles["ReportSection"]),
        ]
    )

    active_weights = dict(weights or DEFAULT_WEIGHTS)
    weight_rows = [["OBJECTIF", "POIDS"]]
    for key, weight in sorted(active_weights.items(), key=lambda item: item[1], reverse=True):
        weight_rows.append([DISPLAY_NAMES.get(key, key), f"{weight*100:.0f} %"])
    weight_table = Table(weight_rows, colWidths=[125 * mm, 35 * mm], repeatRows=1)
    weight_table.setStyle(_table_style())
    story.extend(
        [
            Paragraph(
                "Le score est normalisé entre 0 et 100. La priorité principale "
                "est la réduction de la quantité et du coût du liquide, suivie "
                "de la réduction de la toxicité.",
                styles["ReportBody"],
            ),
            Spacer(1, 3 * mm),
            weight_table,
            Paragraph("Interprétation et limites", styles["ReportSection"]),
            Paragraph(
                "La proposition est issue d’un modèle prédictif exploratoire. "
                "Elle doit être validée par une expérience réelle avant toute "
                "décision industrielle. Les R² des rendements restent faibles dans "
                "la validation croisée actuelle; l'optimiseur peut donc exploiter "
                "une erreur prédictive plutôt qu'un phénomène physique réel. Le coût "
                "énergétique suppose une puissance "
                "constante de 10 kW et un prix de 0,15 $/kWh. Le Toxicity Index est "
                "la somme de HBr, Br2, HCl, Cl2 et HF et ne constitue pas un indice "
                "toxicologique réglementaire.",
                styles["ReportBody"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "Les valeurs après optimisation sont des prédictions de prototype "
                "et non des mesures expérimentales.",
                styles["ReportSmall"],
            ),
        ]
    )

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CDD8D4")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
