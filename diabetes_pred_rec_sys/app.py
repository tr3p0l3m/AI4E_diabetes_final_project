"""Gradio interface for multiclass diabetes risk prediction."""
from __future__ import annotations

from typing import Dict

import gradio as gr

from ml_utils import predict_diabetes


def make_prediction(
    gender: str,
    age: float,
    urea: float,
    cr: float,
    hba1c: float,
    chol: float,
    tg: float,
    hdl: float,
    ldl: float,
    vldl: float,
    bmi: float,
):
    """Run inference and return class probabilities."""
    return predict_diabetes(
        gender=gender,
        age=age,
        urea=urea,
        cr=cr,
        hba1c=hba1c,
        chol=chol,
        tg=tg,
        hdl=hdl,
        ldl=ldl,
        vldl=vldl,
        bmi=bmi,
    )


def build_interface() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(title="Diabetes Risk Classifier") as demo:
        gr.Markdown(
            """
            ## Multiclass Diabetes Classifier
            Provide lab measurements to estimate the likelihood of diabetes classes.
            Make sure the trained model artifact exists at `models/best_model_random_forest.pkl`.
            """
        )

        with gr.Row():
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            age = gr.Slider(0, 120, value=45, step=1, label="Age (years)")
            bmi = gr.Slider(10, 60, value=24, step=0.1, label="BMI")

        with gr.Row():
            urea = gr.Slider(0, 20, value=4.5, step=0.1, label="Urea (mmol/L)")
            cr = gr.Slider(0, 200, value=50, step=1, label="Creatinine (µmol/L)")
            hba1c = gr.Slider(3, 15, value=5.5, step=0.1, label="HbA1c (%)")

        with gr.Row():
            chol = gr.Slider(1, 10, value=4.5, step=0.1, label="Cholesterol (mmol/L)")
            tg = gr.Slider(0, 10, value=1.3, step=0.1, label="Triglycerides (mmol/L)")
            hdl = gr.Slider(0, 5, value=1.1, step=0.1, label="HDL (mmol/L)")

        with gr.Row():
            ldl = gr.Slider(0, 6, value=2.0, step=0.1, label="LDL (mmol/L)")
            vldl = gr.Slider(0, 2, value=0.6, step=0.1, label="VLDL (mmol/L)")

        predict_btn = gr.Button("Predict")
        outputs = gr.JSON(label="Prediction")

        predict_btn.click(
            make_prediction,
            inputs=[gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi],
            outputs=outputs,
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
