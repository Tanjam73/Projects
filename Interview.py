from datetime import datetime

class Candidate:
    def __init__(self, name):
        self.name = name

class Interviewer:
    def __init__(self, name, slots):
        self.name = name
        self.slots = slots

class Scheduler:

    def __init__(self):
        self.schedule = []

    def assign(self, candidates, interviewers):

        candidate_idx = 0

        for interviewer in interviewers:

            for slot in interviewer.slots:

                if candidate_idx >= len(candidates):
                    return

                self.schedule.append({
                    "candidate": candidates[candidate_idx].name,
                    "interviewer": interviewer.name,
                    "slot": slot
                })

                candidate_idx += 1

    def display(self):

        print("\nInterview Schedule\n")

        for interview in self.schedule:

            print(
                f"{interview['candidate']}  -->  "
                f"{interview['interviewer']}  "
                f"({interview['slot']})"
            )

candidates = [
    Candidate("Alice"),
    Candidate("Bob"),
    Candidate("Charlie"),
    Candidate("David"),
    Candidate("Emma")
]

interviewers = [

    Interviewer(
        "HR-1",
        [
            "2026-06-15 10:00",
            "2026-06-15 11:00",
            "2026-06-15 12:00"
        ]
    ),

    Interviewer(
        "Tech-1",
        [
            "2026-06-15 10:00",
            "2026-06-15 11:00",
            "2026-06-15 12:00"
        ]
    )
]

scheduler = Scheduler()

scheduler.assign(
    candidates,
    interviewers
)

scheduler.display()
