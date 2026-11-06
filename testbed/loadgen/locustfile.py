"""Normal traffic plus a healthy traffic burst (the 'traffic spike', which is NOT a fault).

BURST_USERS  users during a burst (baseline is 60). 240 (4x) overloads the single-CPU gateway and breaches the
             SLO, so it is not healthy on this testbed; the default 150 is meant to stay within it.
BURST_FIRST  seconds after start of the first burst; a 150 s burst then repeats every 600 s.
"""
import os

from locust import HttpUser, LoadTestShape, between, task


class Shopper(HttpUser):
    wait_time = between(0.3, 1.2)

    @task(6)
    def browse(self):
        self.client.get("/api/products/%d" % (hash(self) % 5000))

    @task(3)
    def login(self):
        self.client.post("/auth/login")

    @task(2)
    def order(self):
        self.client.post("/api/orders")


class SteadyWithBursts(LoadTestShape):
    base = 60
    burst = int(os.getenv("BURST_USERS", "150"))
    first = int(os.getenv("BURST_FIRST", "120"))
    period, length = 600, 150

    def tick(self):
        t = self.get_run_time()
        in_burst = t >= self.first and ((t - self.first) % self.period) < self.length
        return (self.burst if in_burst else self.base), 20
