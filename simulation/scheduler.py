from typing import List, Callable

class TimeSlotScheduler:
    """Schedule operations for each time slot"""
    
    def __init__(self, slot_duration: float = 1.0):
        self.slot_duration = slot_duration
        self.current_slot = 0
        self.operations: List[Callable] = []
        
    def register_operation(self, operation: Callable):
        """Register operation to execute each slot"""
        self.operations.append(operation)
        
    def execute_slot(self):
        """Execute all operations for current slot"""
        for operation in self.operations:
            operation(self.current_slot)
            
        self.current_slot += 1
        
    def reset(self):
        """Reset scheduler"""
        self.current_slot = 0