"""
AmbuBrain - Data models for the ambulance routing environment.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import random


@dataclass
class Hospital:
    name: str
    zone: str
    total_beds: int
    available_beds: int
    icu_beds: int
    available_icu: int
    staff_on_duty: int
    base_wait_time: float  # minutes
    rating: float = 4.0
    specialty: str = "General"

    def capacity_ratio(self) -> float:
        if self.total_beds == 0:
            return 0.0
        return self.available_beds / self.total_beds

    def icu_ratio(self) -> float:
        if self.icu_beds == 0:
            return 0.0
        return self.available_icu / self.icu_beds

    def effective_wait(self) -> float:
        load = 1.0 - self.capacity_ratio()
        return self.base_wait_time * (1 + load * 2)


@dataclass
class Patient:
    severity: str          # "critical" or "moderate"
    condition_name: str    # "Cardiac Arrest", "Asthma"
    zone: str              # originating zone
    arrival_time: float    # simulation time
    condition_score: float # 0.0 (dying) to 1.0 (stable)

    def survival_probability(self, wait_minutes: float) -> float:
        # Tuned decay to realistic ambulance response times so AI actually succeeds
        decay = 0.015 if self.severity == "critical" else 0.005
        prob = self.condition_score - (decay * wait_minutes)
        return max(0.0, min(1.0, prob))


@dataclass
class TrafficCondition:
    zone_from: str
    zone_to: str
    base_time: float   # minutes
    delay_factor: float  # 1.0 = no delay, 2.0 = double

    def travel_time(self) -> float:
        return self.base_time * self.delay_factor


def create_default_hospitals() -> List[Hospital]:
    return [
        Hospital(
            name="Lilavati Hospital",
            zone="central",
            total_beds=200,
            available_beds=random.randint(20, 80),
            icu_beds=30,
            available_icu=random.randint(2, 15),
            staff_on_duty=random.randint(15, 40),
            base_wait_time=10.0,
            rating=4.9,
            specialty="Cardiology & Trauma"
        ),
        Hospital(
            name="KEM Hospital",
            zone="north",
            total_beds=150,
            available_beds=random.randint(15, 60),
            icu_beds=20,
            available_icu=random.randint(2, 10),
            staff_on_duty=random.randint(10, 30),
            base_wait_time=12.0,
            rating=3.8,
            specialty="General & Burns"
        ),
        Hospital(
            name="Breach Candy Hospital",
            zone="south",
            total_beds=120,
            available_beds=random.randint(10, 50),
            icu_beds=15,
            available_icu=random.randint(1, 8),
            staff_on_duty=random.randint(8, 25),
            base_wait_time=15.0,
            rating=4.7,
            specialty="Neuro & Critical"
        Hospital(
            name="Apollo Hospital",
            zone="east",
            total_beds=180,
            available_beds=random.randint(15, 70),
            icu_beds=25,
            available_icu=random.randint(2, 12),
            staff_on_duty=random.randint(12, 35),
            base_wait_time=13.0,
            rating=4.5,
            specialty="Oncology & General"
        ),
        Hospital(
            name="Fortis Hospital",
            zone="west",
            total_beds=140,
            available_beds=random.randint(10, 55),
            icu_beds=18,
            available_icu=random.randint(1, 9),
            staff_on_duty=random.randint(10, 28),
            base_wait_time=14.0,
            rating=4.2,
            specialty="Orthopedics & Emergency"
        ),
    ]


def create_traffic_conditions(patient_zone: str) -> List[TrafficCondition]:
    zones = ["central", "north", "south", "east", "west"]
    conditions = []
    
    # Introduce random weather impacting traffic
    weather = random.choice(["Clear", "Clear", "Clear", "Heavy Rain", "Fog", "Road Closure"])
    weather_multiplier = 1.0
    if weather == "Heavy Rain":
        weather_multiplier = 1.5
    elif weather == "Fog":
        weather_multiplier = 1.3
    elif weather == "Road Closure":
        weather_multiplier = 1.8

    for zone in zones:
        # Realistic gridlock (1.0x to 1.5x) instead of impossible (2.5x) limits
        delay = round(random.uniform(1.0, 1.5) * weather_multiplier, 2)
        base = 10.0 if zone == patient_zone else 15.0 + random.uniform(0, 10)
        conditions.append(TrafficCondition(
            zone_from=patient_zone,
            zone_to=zone,
            base_time=round(base, 1),
            delay_factor=delay,
        ))
    return conditions, weather


def create_random_patient(sim_time: float = 0.0) -> Patient:
    severity = random.choice(["critical", "moderate", "moderate"])
    cond = ""
    if severity == "critical":
        cond = random.choice(["Cardiac Arrest", "Massive Stroke", "Severe Trauma", "Third-Degree Burns"])
    else:
        cond = random.choice(["Mild Asthma", "Fractured Arm", "Deep Laceration", "High Fever"])
        
    zones = ["central", "north", "south", "east", "west"]
    return Patient(
        severity=severity,
        condition_name=cond,
        zone=random.choice(zones),
        arrival_time=sim_time,
        condition_score=round(random.uniform(0.4, 0.95), 2),
    )
