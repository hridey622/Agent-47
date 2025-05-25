from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

@dataclass
class StepMetrics:
    step_number: int
    timestamp: datetime
    action_type: str
    action_args: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    extracted_content: Optional[str]
    browser_state: Dict[str, Any]
    execution_time: float
    tokens_used: int

class PerformanceTracker:
    def __init__(self, output_dir: str = "performance_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[StepMetrics] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def record_step(self, 
                   step_number: int,
                   action_type: str,
                   action_args: Dict[str, Any],
                   success: bool,
                   error_message: Optional[str],
                   extracted_content: Optional[str],
                   browser_state: Dict[str, Any],
                   execution_time: float,
                   tokens_used: int) -> None:
        """Record metrics for a single step."""
        metrics = StepMetrics(
            step_number=step_number,
            timestamp=datetime.now(),
            action_type=action_type,
            action_args=action_args,
            success=success,
            error_message=error_message,
            extracted_content=extracted_content,
            browser_state=browser_state,
            execution_time=execution_time,
            tokens_used=tokens_used
        )
        self.metrics.append(metrics)
        
    def save_to_json(self) -> None:
        """Save metrics to a JSON file."""
        output_file = self.output_dir / f"performance_log_{self.session_id}.json"
        metrics_dict = [vars(metric) for metric in self.metrics]
        # Convert datetime to string for JSON serialization
        for metric in metrics_dict:
            metric['timestamp'] = metric['timestamp'].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
    def save_to_csv(self) -> None:
        """Save metrics to a CSV file."""
        output_file = self.output_dir / f"performance_log_{self.session_id}.csv"
        df = pd.DataFrame([vars(metric) for metric in self.metrics])
        df.to_csv(output_file, index=False)
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the performance metrics."""
        if not self.metrics:
            return {}
            
        total_steps = len(self.metrics)
        successful_steps = sum(1 for m in self.metrics if m.success)
        failed_steps = total_steps - successful_steps
        
        action_types = {}
        for metric in self.metrics:
            action_type = metric.action_type
            if action_type not in action_types:
                action_types[action_type] = {'total': 0, 'success': 0}
            action_types[action_type]['total'] += 1
            if metric.success:
                action_types[action_type]['success'] += 1
                
        avg_execution_time = sum(m.execution_time for m in self.metrics) / total_steps
        total_tokens = sum(m.tokens_used for m in self.metrics)
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'action_type_stats': action_types,
            'avg_execution_time': avg_execution_time,
            'total_tokens_used': total_tokens
        }
        
    def get_action_success_matrix(self) -> pd.DataFrame:
        """Create a matrix showing success rates for different action types."""
        if not self.metrics:
            return pd.DataFrame()
            
        action_stats = {}
        for metric in self.metrics:
            action_type = metric.action_type
            if action_type not in action_stats:
                action_stats[action_type] = {'success': 0, 'failure': 0}
            if metric.success:
                action_stats[action_type]['success'] += 1
            else:
                action_stats[action_type]['failure'] += 1
                
        df = pd.DataFrame.from_dict(action_stats, orient='index')
        df['total'] = df['success'] + df['failure']
        df['success_rate'] = df['success'] / df['total']
        return df.sort_values('total', ascending=False)
        
    def get_error_analysis(self) -> Dict[str, List[str]]:
        """Analyze error patterns in failed steps."""
        error_patterns = {}
        for metric in self.metrics:
            if not metric.success and metric.error_message:
                error_type = type(metric.error_message).__name__
                if error_type not in error_patterns:
                    error_patterns[error_type] = []
                error_patterns[error_type].append(metric.error_message)
        return error_patterns 