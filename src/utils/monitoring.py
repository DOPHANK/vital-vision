class SystemMonitor:
    """System-wide monitoring and logging."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.log_manager = LogManager()
        self.alert_manager = AlertManager()
        
    def monitor_pipeline(self, pipeline: Any) -> None:
        """Monitor pipeline performance and health."""
        self.metrics_collector.collect(pipeline)
        self.log_manager.log(pipeline)
        self.alert_manager.check_alerts(pipeline)
