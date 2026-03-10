import os
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from core.narrator import narrate
from core.headers import Section

try:
    from weasyprint import HTML
except (ImportError, OSError):
    HTML = None

def generate_terminal_report(state, args):
    """Simplified metric and limitations readout for the terminal."""
    narrate(f"\n{Section.REPORT}")
    
    evaluation = state.evaluation
    detection = state.detection
    if not evaluation or not detection:
        narrate("  [!] Cannot generate terminal report: missing evaluation or detection state.")
        return

    problem_type = detection.get('problem_type', 'unknown')
    
    narrate(f"  Model         : {evaluation.get('best_model_name', 'Unknown')}")
    narrate(f"  Problem Type  : {problem_type.upper()}")
    
    if problem_type == 'classification':
        f1 = evaluation.get('f1_weighted')
        auc = evaluation.get('roc_auc')
        narrate(f"  F1 Weighted   : {f1:.4f}" if f1 is not None else "  F1 Weighted   : N/A")
        narrate(f"  ROC-AUC       : {auc:.4f}" if auc is not None else "  ROC-AUC       : N/A")
        if 'confusion_matrix_inference' in evaluation:
            narrate(f"  Confusion     : {evaluation['confusion_matrix_inference']}")
            
    elif problem_type == 'regression':
        rmse = evaluation.get('rmse')
        r2 = evaluation.get('r2')
        mae = evaluation.get('mae')
        narrate(f"  RMSE          : {rmse:.4f}" if rmse is not None else "  RMSE          : N/A")
        narrate(f"  R²            : {r2:.4f}" if r2 is not None else "  R²            : N/A")
        narrate(f"  MAE           : {mae:.4f}" if mae is not None else "  MAE           : N/A")
        
    elif problem_type == 'clustering':
        sil = evaluation.get('silhouette_score')
        db = evaluation.get('davies_bouldin')
        narrate(f"  Silhouette    : {sil:.4f}" if sil is not None else "  Silhouette    : N/A")
        narrate(f"  Davies-Bouldin: {db:.4f}" if db is not None else "  Davies-Bouldin: N/A")

    limitations = evaluation.get('limitations', [])
    narrate(f"\n  Limitations ({len(limitations)}):")
    for i, lim in enumerate(limitations, 1):
        narrate(f"    {i}. {lim}")
        
    narrate(f"\n  Model saved   : {evaluation.get('model_path', 'Not saved')}")

def render_confusion_matrix(cm, class_labels):
    """Renders a Seaborn heatmap from a numpy Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='g', cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def generate_pdf_report(state, args):
    """Compiles the full artifact suite into a professional PDF layout using weasyprint."""
    if HTML is None:
        narrate("  [!] weasyprint is not installed. PDF generation aborted. Use --report terminal.")
        return
        
    evaluation = state.evaluation
    if not evaluation:
        narrate("  [!] Cannot generate PDF report: missing evaluation state.")
        return
        
    narrate("  -> Compiling PDF artifacts...")
    
    # 1. Prepare template variables
    template_dir = str(Path(__file__).parent.resolve())
    env = Environment(loader=FileSystemLoader(template_dir))
    try:
        template = env.get_template("report_template.html")
    except Exception as e:
        narrate(f"  [!] Missing report_template.html in {template_dir}: {e}")
        return
        
    shap_plots = {}
    cm_plot = None
    temp_files = []
    
    try:
        # 2. Render Confusion Matrix Heatmap
        cm_array = evaluation.get('confusion_matrix')
        if cm_array is not None and isinstance(cm_array, np.ndarray):
            class_labels = state.detection.get('class_labels', [])
            if not class_labels:
                class_labels = [str(i) for i in range(len(cm_array))]
                
            fig_cm = render_confusion_matrix(cm_array, class_labels)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig_cm.savefig(f.name, bbox_inches='tight', dpi=150)
                cm_plot = f.name
                temp_files.append(f.name)
            plt.close(fig_cm)
            
        # 3. Buffer SHAP plots
        shap_summary = evaluation.get('shap_summary_plot')
        if shap_summary is not None:
             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                 shap_summary.savefig(f.name, bbox_inches='tight', dpi=150)
                 shap_plots['summary'] = f.name
                 temp_files.append(f.name)
                 
        shap_waterfall = evaluation.get('shap_waterfall_plot')
        if shap_waterfall is not None:
             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                 shap_waterfall.savefig(f.name, bbox_inches='tight', dpi=150)
                 shap_plots['waterfall'] = f.name
                 temp_files.append(f.name)
                 
        # Support clustering visualization placeholder (PCA)
        cluster_viz = evaluation.get('cluster_visualization')
        cluster_plot = None
        if cluster_viz is not None:
             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                 cluster_viz.savefig(f.name, bbox_inches='tight', dpi=150)
                 cluster_plot = f.name
                 temp_files.append(f.name)

        # Support regression residual plot
        residual_fig = evaluation.get('residual_plot')
        residual_plot = None
        if residual_fig is not None:
             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                 residual_fig.savefig(f.name, bbox_inches='tight', dpi=150)
                 residual_plot = f.name
                 temp_files.append(f.name)

        # 4. Inject Jinja Template
        timestamp = state.completed_at or state.started_at
        
        rendered_html = template.render(
            evaluation=evaluation,
            audit=state.audit,
            detection=state.detection,
            args=state.args,
            timestamp=timestamp,
            cm_plot=cm_plot,
            shap_plots=shap_plots,
            cluster_plot=cluster_plot,
            residual_plot=residual_plot,
            feature_result=state.feature_result
        )
        
        # 5. Export PDF
        timestamp_safe = str(timestamp).replace(':', '').replace('-', '').replace(' ', '_')
        output_path = os.path.join(args.out, "reports", f"report_{timestamp_safe}.pdf")
        HTML(string=rendered_html).write_pdf(output_path)
        narrate(f"  [+] PDF generated successfully at {output_path}")

    except Exception as e:
        narrate(f"  [!] Failed to compile PDF report: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Guarantee Destructor Cleanup
        for f_path in temp_files:
            try:
                os.unlink(f_path)
            except Exception:
                pass
