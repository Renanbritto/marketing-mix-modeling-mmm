"""Command-Line Interface (CLI) and execution entrypoint for Marketing Mix Modeling."""

import argparse
import sys
import os
import json
import uvicorn
import pytest

from src.config.settings import settings
from src.data.sample_generator import MarketingDataGenerator
from src.engine.mmm_model import MarketingMixModel
from src.engine.optimizer import BudgetOptimizer
from src.domain.models import CHANNEL_DISPLAY_NAMES
from api.index import app


def run_pipeline():
    """Execute end-to-end econometric modeling and budget optimization pipeline."""
    print("================================================================================")
    print(" MARKETING MIX MODELING (MMM) & MULTI-CHANNEL BUDGET OPTIMIZER")
    print("================================================================================")
    print(f"Versao: {settings.app_version} | Ambiente: {settings.environment}")
    print("Iniciando pipeline econometrico de atribuicao multicanal e otimizacao...")

    # 1. Ingest / Generate Data
    print(f"\n[1/5] Gerando serie temporal de investimento e receita ({settings.n_weeks} semanas)...")
    generator = MarketingDataGenerator(seed=settings.random_seed)
    df = generator.generate(n_weeks=settings.n_weeks)
    total_obs = len(df)
    total_rev = df["revenue"].sum()
    print(f"      [OK] {total_obs} semanas processadas. Receita Total: R$ {total_rev:,.2f}")

    # 2. Fit Econometric Model (Adstock + Hill + Ridge)
    print("\n[2/5] Treinando motor econometrico (Adstock carryover + Saturacao Hill + Ridge)...")
    model = MarketingMixModel(alpha=1.0)
    model.fit(df)
    metrics = model.get_metrics()
    print(f"      [OK] Modelo ajustado com sucesso.")
    print(f"           R2 Score:           {metrics['r_squared']:.4f}")
    print(f"           R2 Ajustado:        {metrics['adjusted_r_squared']:.4f}")
    print(f"           MAPE:               {metrics['mape_pct']:.2f}%")
    print(f"           MAE:                R$ {metrics['mae']:,.2f}")
    print(f"           NRMSE:              {metrics['nrmse_pct']:.2f}%")

    # 3. Revenue Decomposition
    print("\n[3/5] Calculando decomposicao de receita (Base Organica vs. Midia vs. Sazonalidade)...")
    decomp = model.decompose()
    print("--------------------------------------------------------------------------------")
    print(f" {'Canal':<32} | {'Investimento':<15} | {'Receita Atrib.':<16} | {'ROAS':<8} | {'Contrib %':<9}")
    print("--------------------------------------------------------------------------------")
    
    total_spend_all = 0.0
    for ch, params in settings.channels.items():
        ch_spend = df[f"spend_{ch}"].sum()
        total_spend_all += ch_spend
        ch_rev = decomp["channel_contributions"].get(ch, 0.0)
        ch_roas = ch_rev / ch_spend if ch_spend > 0 else 0.0
        ch_pct = decomp["channel_contribution_pcts"].get(ch, 0.0)
        display = CHANNEL_DISPLAY_NAMES.get(ch, ch)
        print(f" {display:<32} | R$ {ch_spend:>11,.0f} | R$ {ch_rev:>12,.0f} | {ch_roas:>6.2f}x | {ch_pct:>7.1f}%")

    print("--------------------------------------------------------------------------------")
    base_rev = decomp["base_organic_total"]
    base_pct = decomp["base_organic_pct"]
    print(f" {'Baseline Organico (Sem Midia)':<32} | {'-':<15} | R$ {base_rev:>12,.0f} | {'-':<8} | {base_pct:>7.1f}%")
    print("--------------------------------------------------------------------------------")

    # 4. Multi-Channel Budget Optimization
    current_weekly_budget = sum(p.weekly_budget_default for p in settings.channels.values())
    print(f"\n[4/5] Executando otimizacao matematica com SciPy SLSQP (Budget: R$ {current_weekly_budget:,.2f}/sem)...")
    optimizer = BudgetOptimizer(model)
    opt_result = optimizer.optimize(
        total_budget=current_weekly_budget,
        min_per_channel={"crm_email": 2_000.0},
        max_per_channel={"meta_ads": 60_000.0},
    )

    print("--------------------------------------------------------------------------------")
    print(f" {'Canal':<32} | {'Atual / Sem':<14} | {'Otimo / Sem':<14} | {'Variacao':<10} | {'Status'}")
    print("--------------------------------------------------------------------------------")
    for c in opt_result["channels"]:
        ch = c["channel"]
        cur = settings.channels[ch].weekly_budget_default
        opt = c["spend"]
        diff_pct = ((opt - cur) / cur * 100) if cur > 0 else 0.0
        diff_str = f"{diff_pct:+.1f}%"
        display = CHANNEL_DISPLAY_NAMES.get(ch, ch)
        print(f" {display:<32} | R$ {cur:>10,.0f} | R$ {opt:>10,.0f} | {diff_str:>9} | {c['health']}")

    print("--------------------------------------------------------------------------------")
    print(f" Receita Semanal Projetada (Alocacao Otima): R$ {opt_result['projected_revenue']:,.2f}")
    print(f" ROAS Global Projetado:                      {opt_result['roas_global']:.2f}x")
    print(f" Lift Estimado de Receita:                   +{opt_result['lift_vs_uniform_pct']:.2f}%")
    print("--------------------------------------------------------------------------------")

    # 5. Export JSON Report
    os.makedirs("output", exist_ok=True)
    report_path = "output/mmm_report.json"
    report_data = {
        "metrics": metrics,
        "decomposition_summary": {
            "total_revenue": decomp["total_actual_revenue"],
            "base_organic": decomp["base_organic_total"],
            "base_organic_pct": decomp["base_organic_pct"],
            "channel_contributions": decomp["channel_contributions"],
            "channel_contribution_pcts": decomp["channel_contribution_pcts"],
        },
        "optimization": opt_result,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[5/5] Relatorio analitico exportado: {os.path.abspath(report_path)}")
    print("================================================================================")
    print(" Pipeline econometrico executado com sucesso.")
    print("================================================================================")


def serve_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI HTTP and Dashboard server."""
    print(f"Iniciando API MMM em http://{host}:{port}...")
    uvicorn.run("src.api.server:create_app", host=host, port=port, factory=True, reload=False)


def run_tests():
    """Run full test suite via Pytest."""
    print("Executando suite de testes automatizados com Pytest...")
    ret = pytest.main(["-v", "tests/"])
    sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(description="Marketing Mix Modeling (MMM) & Budget Optimizer")
    parser.add_argument("--run-pipeline", action="store_true", help="Executar pipeline econometrico completo")
    parser.add_argument("--serve-api", action="store_true", help="Iniciar servidor FastAPI com dashboard")
    parser.add_argument("--run-tests", action="store_true", help="Executar testes unitarios e de integracao")
    parser.add_argument("--host", default="0.0.0.0", help="Host para a API")
    parser.add_argument("--port", type=int, default=8000, help="Porta para a API")

    args = parser.parse_args()

    if args.serve_api:
        serve_api(host=args.host, port=args.port)
    elif args.run_tests:
        run_tests()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
