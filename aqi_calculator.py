import pandas as pd
import numpy as np

class AQICalculator:
    def __init__(self):
        self.iaqi_breakpoints = [0, 50, 100, 150, 200, 300, 400, 500]
        self.pollutant_breakpoints = {
            'SO2_24h': [0, 50, 150, 475, 800, 1600, 2100, 2620],
            'NO2_24h': [0, 40, 80, 180, 280, 565, 750, 940],
            'PM10_24h': [0, 50, 150, 250, 350, 420, 500, 600],
            'CO_24h': [0, 2, 4, 14, 24, 36, 48, 60],
            'O3_8h': [0, 100, 160, 215, 265, 800, 1000, 1200],
            'PM2.5_24h': [0, 35, 75, 115, 150, 250, 350, 500]
        }
    
    def calculate_iaqi(self, pollutant, concentration):
        if pollutant not in self.pollutant_breakpoints or concentration < 0:
            return 500.000
        
        breakpoints = self.pollutant_breakpoints[pollutant]
        
        for i in range(len(breakpoints) - 1):
            if concentration <= breakpoints[i + 1]:
                bp_lo = breakpoints[i]
                bp_hi = breakpoints[i + 1]
                iaqi_lo = self.iaqi_breakpoints[i]
                iaqi_hi = self.iaqi_breakpoints[i + 1]
                
                if bp_hi == bp_lo:
                    iaqi = iaqi_lo
                else:
                    iaqi = (iaqi_hi - iaqi_lo) / (bp_hi - bp_lo) * (concentration - bp_lo) + iaqi_lo
                
                return round(iaqi, 3)
        
        return 500.000
    
    def calculate_aqi(self, pollutant_data):
        iaqis = {}
        for pollutant, concentration in pollutant_data.items():
            if concentration is not None and concentration >= 0:
                iaqi = self.calculate_iaqi(pollutant, concentration)
                iaqis[pollutant] = iaqi
        
        if not iaqis:
            return None
        
        aqi = max(iaqis.values())
        return round(aqi, 3)

def main():
    calculator = AQICalculator()
    
    while True:
        try:
            pm25 = input("PM2.5: ")
            pm10 = input("PM10: ")
            o3 = input("O3: ")
            so2 = input("SO2: ")
            no2 = input("NO2: ")
            co = input("CO: ")
            
            pollutant_data = {}
            if pm25: pollutant_data['PM2.5_24h'] = float(pm25)
            if pm10: pollutant_data['PM10_24h'] = float(pm10)
            if o3: pollutant_data['O3_8h'] = float(o3)
            if so2: pollutant_data['SO2_24h'] = float(so2)
            if no2: pollutant_data['NO2_24h'] = float(no2)
            if co: pollutant_data['CO_24h'] = float(co)
            
            if pollutant_data:
                aqi = calculator.calculate_aqi(pollutant_data)
                print(f"AQI: {aqi:.3f}")
            
            if input("继续? (y/n): ").lower() != 'y':
                break
                
        except (ValueError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main() 