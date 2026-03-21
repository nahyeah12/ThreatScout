import random


def add_noise(value, scale=0.05):
    """Small variation around baseline"""
    return max(0, min(1, value + random.uniform(-scale, scale)))


def simulate_normal_behaviour(employee):
    """Update behaviour for normal employees"""
    new_behaviour = {}

    for key, base_value in employee.baseline.items():
        new_behaviour[key] = add_noise(base_value)

    return new_behaviour


def simulate_insider_behaviour(employee, step):
    """
    Insider gradually deviates from baseline
    step = time step (used to increase severity)
    """
    new_behaviour = {}

    for key, base_value in employee.baseline.items():
        new_behaviour[key] = add_noise(base_value)

    # Gradual escalation
    factor = min(1.0, step * 0.1)

    new_behaviour["after_hours_login"] += 0.3 * factor
    new_behaviour["file_access"] += 0.4 * factor
    new_behaviour["usb_usage"] += 0.5 * factor
    new_behaviour["job_sites"] += 0.3 * factor

    # clamp values to [0,1]
    for key in new_behaviour:
        new_behaviour[key] = min(1.0, new_behaviour[key])

    return new_behaviour


def update_behaviour(employees, step):
    """
    Update all employees for one timestep
    """
    for emp in employees:
        if emp.is_insider:
            emp.current = simulate_insider_behaviour(emp, step)
        else:
            emp.current = simulate_normal_behaviour(emp)


def inject_insider(employees):
    """Randomly select one employee as insider"""
    emp = random.choice(employees)
    emp.is_insider = True
    return emp