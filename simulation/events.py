from dataclasses import dataclass
from typing import Any, Callable
from queue import PriorityQueue

@dataclass
class Event:
    """Simulation event"""
    time: float
    type: str
    data: Any
    callback: Callable = None
    
    def __lt__(self, other):
        return self.time < other.time

class EventManager:
    """Manage simulation events"""
    
    def __init__(self):
        self.event_queue = PriorityQueue()
        self.current_time = 0
        self.event_handlers = {}
        
    def schedule_event(self, event: Event):
        """Schedule an event"""
        self.event_queue.put(event)
        
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type] = handler
        
    def process_events(self, until_time: float):
        """Process events until specified time"""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            
            if event.time > until_time:
                # Put it back and stop
                self.event_queue.put(event)
                break
                
            self.current_time = event.time
            
            # Execute callback or handler
            if event.callback:
                event.callback(event.data)
            elif event.type in self.event_handlers:
                self.event_handlers[event.type](event.data)