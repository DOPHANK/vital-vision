class TestFramework:
    """Comprehensive testing framework."""
    
    def __init__(self):
        self.unit_tester = UnitTester()
        self.integration_tester = IntegrationTester()
        self.performance_tester = PerformanceTester()
        
    def run_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        results = {
            'unit_tests': self.unit_tester.run(),
            'integration_tests': self.integration_tester.run(),
            'performance_tests': self.performance_tester.run()
        }
        return results
