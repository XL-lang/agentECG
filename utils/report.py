from typing import List,Dict
from data.EcgSignals import EcgSignals
from models.pytorch_inception.classifier import ECGSignalClassifier

class ECGQA_ptbxl_report:
    def __init__(self, ecg_signals: Dict[str, EcgSignals]) -> None:
        self.ecg_signals = ecg_signals

    def report(self) -> Dict[str, str]:
        reports = {}
        for ecg_name, ecg_signal in self.ecg_signals.items():
            classifier = ECGSignalClassifier(ecg_signal)
            reports[ecg_name] = classifier.report_predict_result()
        return reports