#!/usr/bin/env python3
"""
evaluator.py — ADAS Performance Analysis & Reporting Engine.

This node acts as a black-box monitors for the simulation. It subscribes 
to safety metrics (TTC), alert signals from both Fixed and ML ADAS, 
and ground-truth simulation state to quantify system performance.

Metrics Calculated:
- Classification Accuracy: Precision, Recall, F1 for "Dangerous Driving" detection.
- Statistical Distributions: ROC Curves and AUC (Area Under Curve).
- False Positive Analysis: Quantitative comparison of nuisance alerts.
- Safety Physics: Cumulative time spent in "Close Call" (<2.5m) or 
  "Collision" (<1.0m) states.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import LaserScan
import math
import time, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless plotting for server/container environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)


class Evaluator(Node):
    """
    ROS2 Node for real-time data logging and post-run statistical analysis.
    """
    def __init__(self):
        super().__init__('evaluator')

        # ─── Telemetry Subscriptions ───
        self.create_subscription(Bool, '/adas/alert/fixed', self.fixed_cb, 10)
        self.create_subscription(Bool, '/adas/alert/ml', self.ml_cb, 10)
        self.create_subscription(String, '/active_policy', self.policy_cb, 10)
        self.create_subscription(Float32, '/adas/ttc/fixed', self.ttc_fixed_cb, 10)
        self.create_subscription(Float32, '/adas/ttc/ml', self.ttc_ml_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # ─── Internal Buffers ───
        self.fixed_alert = False
        self.ml_alert    = False
        self.current_policy = 'normal'
        self.ttc_fixed   = 999.0
        self.ttc_ml      = 999.0
        self.min_distance = 100.0

        self.results     = []
        self.start_time  = time.time()
        self.last_policy_change = time.time()
        self.fixed_alert_times = []
        self.ml_alert_times    = []

        # Heartbeat: Record synchronized state at 2Hz
        self.timer = self.create_timer(0.5, self.record_state)
        self.get_logger().info('Performance Evaluator Online. Logging active.')

    # ── Callbacks ──────────────────────────────────────────────────────

    def fixed_cb(self, msg):
        """Monitors classical ADAS alert triggers."""
        self.fixed_alert = msg.data
        if msg.data:
            self.fixed_alert_times.append(time.time() - self.start_time)

    def ml_cb(self, msg):
        """Monitors ML-Adaptive ADAS alert triggers."""
        self.ml_alert = msg.data
        if msg.data:
            self.ml_alert_times.append(time.time() - self.start_time)

    def policy_cb(self, msg):
        """Tracks the ground-truth driving style (e.g., Aggressive vs Defensive)."""
        if msg.data != self.current_policy:
            self.last_policy_change = time.time()
        self.current_policy = msg.data

    def ttc_fixed_cb(self, msg):
        """Logs standard Time-to-Collision calculations."""
        self.ttc_fixed = msg.data

    def ttc_ml_cb(self, msg):
        """Logs ML-adjusted (Adaptive) Time-to-Collision values."""
        self.ttc_ml = msg.data

    def scan_cb(self, msg):
        """Extracts the minimum distance in the frontal corridor for safety analysis."""
        mid = len(msg.ranges) // 2
        # Use +/- 15 degree sector around center
        vals = [r for r in msg.ranges[mid-15:mid+15] if not math.isnan(r) and not math.isinf(r) and r > 0.1]
        self.min_distance = min(vals) if vals else 100.0

    # ── Data Logging ───────────────────────────────────────────────────

    def record_state(self):
        """Synchronizes and stores current node state into the results buffer."""
        t = time.time() - self.start_time
        # Ground Truth: Aggressive policies are treated as "Dangerous" scenarios
        dangerous = 1 if self.current_policy in ['aggressive', 'late_braking'] else 0

        self.results.append({
            'time': round(t, 2),
            'policy': self.current_policy,
            'is_dangerous': dangerous,
            'fixed_alert': 1 if self.fixed_alert else 0,
            'ml_alert': 1 if self.ml_alert else 0,
            'ttc_fixed': round(self.ttc_fixed, 2),
            'ttc_ml': round(self.ttc_ml, 2),
            'min_dist': round(self.min_distance, 2)
        })

    # ── Final Statistical Analysis ─────────────────────────────────────

    def generate_report(self):
        """
        Processes buffers and exports structured data + visualization.
        Called on KeyboardInterrupt (Shutdown).
        """
        if len(self.results) < 10:
            self.get_logger().warn('Insufficient data for a meaningful report.')
            return

        os.makedirs('evaluation', exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv('evaluation/adas_performance_raw.csv', index=False)
        self.get_logger().info(f'Logged {len(df)} samples to CSV.')

        y_true  = df['is_dangerous'].values
        y_fixed = df['fixed_alert'].values
        y_ml    = df['ml_alert'].values

        # ── 1. Classification Performance Table ──
        metrics = {}
        for name, y_pred in [('Fixed', y_fixed), ('ML', y_ml)]:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f = f1_score(y_true, y_pred, zero_division=0)
            a = accuracy_score(y_true, y_pred)
            metrics[name] = {'Precision': p, 'Recall': r, 'F1': f, 'Accuracy': a}
            self.get_logger().info(f'[METRICS] {name} System: P={p:.2f}, R={r:.2f}, F1={f:.2f}')

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv('evaluation/metrics_summary.csv')
        
        # ── 2. Structural Safety Metrics ──
        min_dists = df['min_dist'].values
        # Cumulative duration spent in critical distance thresholds (0.5s sampling)
        time_crashed  = (min_dists < 1.0).sum() * 0.5
        time_close    = ((min_dists >= 1.0) & (min_dists < 2.5)).sum() * 0.5
        
        with open('evaluation/safety_exposure_report.txt', 'w') as f:
            f.write("=== ADAS SAFETY EXPOSURE REPORT ===\n")
            f.write(f"Collision Dwell Time (<1.0m): {time_crashed:.1f}s\n")
            f.write(f"Close-Call Dwell Time (<2.5m): {time_close:.1f}s\n")

        # ── 3. ROC Characteristics ──
        try:
            fpr_f, tpr_f, _ = roc_curve(y_true, y_fixed)
            fpr_m, tpr_m, _ = roc_curve(y_true, y_ml)
            plt.figure(figsize=(10, 7))
            plt.plot(fpr_f, tpr_f, 'o-', label=f'Traditional Fixed (AUC={auc(fpr_f,tpr_f):.2f})')
            plt.plot(fpr_m, tpr_m, 'x-', label=f'ML-Adaptive (AUC={auc(fpr_m,tpr_m):.2f})')
            plt.plot([0,1],[0,1],'k--', alpha=0.3)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('ROC Curve: ADAS Detection Sensitivity')
            plt.legend(); plt.grid(True, linestyle='--')
            plt.savefig('evaluation/performance_roc.png', dpi=150)
            plt.close()
        except: pass

        # ── 4. FPR Comparison (Bar Chart) ──
        try:
            cm_f = confusion_matrix(y_true, y_fixed, labels=[0,1])
            cm_m = confusion_matrix(y_true, y_ml, labels=[0,1])
            if cm_f.shape == (2,2) and cm_m.shape == (2,2):
                fpr_s = cm_f[0,1] / (cm_f[0,0] + cm_f[0,1] + 1e-9)
                fpr_a = cm_m[0,1] / (cm_m[0,0] + cm_m[0,1] + 1e-9)
                plt.figure(figsize=(7,5))
                sns.barplot(x=['Traditional Fixed', 'ML-Adaptive'], y=[fpr_s, fpr_a], palette='Spectral')
                plt.title('Nuisance Alert Rate (FPR) Comparison')
                plt.ylabel('FPR (Lower is Better)')
                plt.savefig('evaluation/nuisance_alerts_fpr.png', dpi=150)
                plt.close()
        except: pass

        # ── 5. Confusion Matrix (ML Visualizer) ──
        try:
            cm = confusion_matrix(y_true, y_ml, labels=[0,1])
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                        xticklabels=['Safe','Dangerous'], yticklabels=['Safe','Dangerous'])
            plt.title('Confusion Matrix: ML ADAS Decisions')
            plt.ylabel('Ground Truth Intent'); plt.xlabel('ADAS System Action')
            plt.tight_layout()
            plt.savefig('evaluation/confusion_matrix_ml.png', dpi=150)
            plt.close()
        except: pass

        self.get_logger().info('Post-run evaluation complete. Files saved to /evaluation.')


def main(args=None):
    rclpy.init(args=args)
    node = Evaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Generate report only on clean exit
        node.generate_report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
