import requests
import json
import pandas as pd

# API URL (Backend running locally)
API_URL = "http://localhost:8000/api/model/predict-params"

# Define 5 exoplanets (real ones, not necessarily in the training set)
# We assume "confirmed" status implies Flags are all 0.
# Data is approximate for testing purposes.

test_planets = [
    {
        "name": "TOI-700 d (Habitable Zone Earth-size)",
        "features": {
            "orbital_period_days": 37.42,
            "transit_duration_hours": 2.5,  # smooth estimate
            "transit_depth_ppm": 1000.0,    # estimate
            "planet_radius_earth": 1.19,
            "equilibrium_temperature_K": 269.0,
            "insolation_flux_Earth": 0.86,
            "stellar_radius_solar": 0.42,
            "stellar_temperature_K": 3480.0,
            "signal_to_noise": 20.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0
        }
    },
    {
        "name": "K2-18 b (Hycean World Candidate)",
        "features": {
            "orbital_period_days": 32.9,
            "transit_duration_hours": 3.2,
            "transit_depth_ppm": 2900.0,
            "planet_radius_earth": 2.6,
            "equilibrium_temperature_K": 265.0,
            "insolation_flux_Earth": 0.94,
            "stellar_radius_solar": 0.41,
            "stellar_temperature_K": 3457.0,
            "signal_to_noise": 40.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0
        }
    },
    {
        "name": "WASP-121 b (Hot Jupiter)",
        "features": {
            "orbital_period_days": 1.27,
            "transit_duration_hours": 2.9,
            "transit_depth_ppm": 16000.0, # 1.6%
            "planet_radius_earth": 20.0,  # ~1.8 R_jup
            "equilibrium_temperature_K": 2500.0,
            "insolation_flux_Earth": 500.0, # High
            "stellar_radius_solar": 1.46,
            "stellar_temperature_K": 6460.0,
            "signal_to_noise": 200.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0
        }
    },
    {
        "name": "TRAPPIST-1 e (Earth-like)",
        "features": {
            "orbital_period_days": 6.1,
            "transit_duration_hours": 1.0,
            "transit_depth_ppm": 5000.0,
            "planet_radius_earth": 0.92,
            "equilibrium_temperature_K": 251.0,
            "insolation_flux_Earth": 0.66,
            "stellar_radius_solar": 0.12,
            "stellar_temperature_K": 2550.0,
            "signal_to_noise": 30.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0
        }
    },
    {
        "name": "CoRoT-7 b (Lava World)",
        "features": {
            "orbital_period_days": 0.85,
            "transit_duration_hours": 1.3,
            "transit_depth_ppm": 350.0,
            "planet_radius_earth": 1.58,
            "equilibrium_temperature_K": 1750.0,
            "insolation_flux_Earth": 300.0,
            "stellar_radius_solar": 0.87,
            "stellar_temperature_K": 5275.0,
            "signal_to_noise": 45.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0
        }
    },
     {
        "name": "False Positive Scenarion (High Ephemeris Match)",
        "features": {
            "orbital_period_days": 10.0,
            "transit_duration_hours": 4.0,
            "transit_depth_ppm": 1000.0,
            "planet_radius_earth": 2.0,
            "equilibrium_temperature_K": 600.0,
            "insolation_flux_Earth": 10.0,
            "stellar_radius_solar": 1.0,
            "stellar_temperature_K": 5778.0,
            "signal_to_noise": 50.0,
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 1 # Flagged!
        }
    }
]

def run_tests():
    print(f"📡 Connecting to {API_URL}...")
    
    # Prepare payload
    items = []
    for i, p in enumerate(test_planets):
        item = p["features"].copy()
        item["dataset"] = "test_custom"
        item["object_id"] = p["name"]
        items.append(item)
    
    payload = {"items": items}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        predictions = data["predictions"]
        
        print("\n🔎 --- RESULTS ---")
        for i, pred in enumerate(predictions):
            name = pred["object_id"]
            score = pred["score_confirmed"]
            is_confirmed = pred["pred_confirmed"] == 1
            status_icon = "✅ REF" if is_confirmed else "❌ FP"
            
            print(f"\nExample {i+1}: {name}")
            print(f"   Status: {status_icon} (Score: {score:.4f})")
            
            # Simple validation check (we expect first 5 to be confirmed, last one to be FP)
            expected = True if i < 5 else False
            if is_confirmed == expected:
                 print("   MATCH: Correct prediction")
            else:
                 print("   MISMATCH: Unexpected result")

    except Exception as e:
        print(f"❌ Error during request: {e}")
        if 'response' in locals():
            print(response.text)

if __name__ == "__main__":
    run_tests()
