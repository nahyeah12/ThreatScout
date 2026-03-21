import random
import uuid

# Define roles
ROLES = ["Engineer", "HR", "Finance", "Admin"]

# Baseline behavior per role
ROLE_BASELINES = {
    "Engineer": {
        "after_hours_login": 0.2,
        "file_access": 0.5,
        "usb_usage": 0.1,
        "job_sites": 0.05
    },
    "HR": {
        "after_hours_login": 0.1,
        "file_access": 0.2,
        "usb_usage": 0.05,
        "job_sites": 0.02
    },
    "Finance": {
        "after_hours_login": 0.15,
        "file_access": 0.7,
        "usb_usage": 0.1,
        "job_sites": 0.03
    },
    "Admin": {
        "after_hours_login": 0.25,
        "file_access": 0.6,
        "usb_usage": 0.2,
        "job_sites": 0.02
    }
}


class Employee:
    def __init__(self, role):
        self.id = str(uuid.uuid4())[:6]
        self.role = role

        # baseline behavior (fixed)
        self.baseline = ROLE_BASELINES[role].copy()

        # current behavior (changes over time)
        self.current = self.baseline.copy()

        # risk tracking
        self.risk_score = 0.0
        self.risk_label = "Low"
        self.risk_history = []

        # anomaly score placeholder
        self.anomaly_score = 0.0

        # insider flag
        self.is_insider = False

    def __repr__(self):
        return f"{self.id} | {self.role} | Risk: {self.risk_label}"


def create_employees(n=20):
    employees = []
    for _ in range(n):
        role = random.choice(ROLES)
        emp = Employee(role)
        employees.append(emp)
    return employees