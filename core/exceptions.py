class ContractViolationError(Exception):
    """
    Raised when an inter-module interface contract is not met.
    This indicates a disagreement between what one module promises
    to return and what another module expects to receive.
    Always fatal — never catch and continue.
    """
    pass

class PipelineStepError(Exception):
    """
    Raised when a pipeline step fails during execution.
    Contains the step name and original exception for clean reporting.
    """
    def __init__(self, step_name, original_error):
        self.step_name = step_name
        self.original_error = original_error
        super().__init__(f"Pipeline failed at step '{step_name}': {original_error}")
