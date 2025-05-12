class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
        self.recovery_manager = RecoveryManager()
        self.alert_system = AlertSystem()
        
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle errors with recovery mechanisms."""
        self.error_logger.log(error, context)
        self.recovery_manager.attempt_recovery(error, context)
        self.alert_system.notify(error, context)
