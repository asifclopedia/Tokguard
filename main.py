import sys
from PyQt6.QtWidgets import QApplication
from analyzer import PromptAnalyzer
from telemetry import TelemetryStore
from ui_widget import TokguardWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Tokguard")

    store = TelemetryStore("tokguard.db")
    settings = store.get_settings()

    analyzer = PromptAnalyzer(
        encoding_name=settings.get("encoding", "cl100k_base"),
        session_budget=int(settings.get("session_budget", "12000") or "12000"),
        project_name=settings.get("project_name", "Tokguard"),
        project_description=settings.get("project_description", ""),
        model_description=settings.get("model_description", ""),
        rewriter_endpoint=settings.get("rewriter_endpoint", ""),
        rewriter_model=settings.get("rewriter_model", ""),
        rewriter_api_key=settings.get("rewriter_api_key", ""),
    )

    widget = TokguardWidget(store=store, analyzer=analyzer)
    widget.show()

    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
