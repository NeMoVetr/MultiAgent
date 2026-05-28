import logging

import geocoder
import numpy as np
import pandas as pd
import pyet
import requests

logger = logging.getLogger(__name__)


class Irrigation:
    """
    Класс Irrigation Agent — реализует расчётную модель управляющего решения
    по дефициту влаги корнеобитаемого слоя (FAO-56).
    """

    # Приватные константы класса (name mangling)
    __theta_fc = 0.24
    __theta_wp = 0.0827
    __Zr = 0.5
    __p_tab = 0.50
    __Kc = 1.05
    __u_max = 5.0

    def __init__(
            self,
            soil_raw: list[float],  # Показания 4 зондов TR-4H01X (%)
            T_mean: float,  # °C
            RH_mean: float,  # %
            wind_speed: float,  # м/с
            pressure_hpa: float,  # hPa
            solar_radiation_wm2: float,  # Вт/м²
            rain_mm: float = 0.0,  # мм

    ):
        """
        Конструктор принимает данные с датчиков и сразу выполняет расчёт.
        """
        self.soil_raw = soil_raw
        self.T_mean = T_mean
        self.RH_mean = RH_mean
        self.wind_speed = wind_speed
        self.pressure_hpa = pressure_hpa
        self.solar_radiation_wm2 = solar_radiation_wm2
        self.rain_mm = rain_mm

        self.lat, self.lng = self.__get_location()
        self.elevation = self.__get_elevation(self.lat, self.lng)

    @staticmethod
    def __get_location() -> tuple[float, float]:
        """Автоматическое определение широты и долготы по IP"""
        try:
            g = geocoder.ip('me')
            if g.ok and g.lat is not None and g.lng is not None:
                logger.info(f"Determined location: lat={g.lat}, lng={g.lng}")
                return float(g.lat), float(g.lng)
            else:
                raise RuntimeError(
                    "It was not possible to get the coordinates. "
                    "Check your internet connection"
                )
        except Exception:
            raise RuntimeError(
                "It was not possible to get the coordinates. "
                "Check your internet connection"
            )

    @staticmethod
    def __get_elevation(lat: float, lng: float) -> float:
        """Автоматическое определение высоты над уровнем моря (земная поверхность)"""
        try:
            # Используем бесплатный Open-Elevation API
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lng}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                elevation = data['results'][0]['elevation']
                logger.info(f"Determined elevation: {elevation} meters")

                return float(elevation)
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            raise RuntimeError(
                "Couldn't determine height by coordinates."
                "Check your internet connection"
            )

    @classmethod
    def _get_const(cls, name: str):
        """Вспомогательный метод для доступа к приватным константам"""
        mangled_name = f"_{cls.__name__}__{name}"
        return getattr(cls, mangled_name)

    def __calculate(self) -> bool:
        """Приватный метод расчёта (внутренняя реализация)"""

        # =========================================================
        # 1. ПРЕОБРАЗОВАНИЕ ВЛАЖНОСТИ ПОЧВЫ
        # =========================================================
        theta_values = np.array(self.soil_raw) / 100.0
        theta_avg = np.mean(theta_values)

        theta_fc = self._get_const("theta_fc")
        theta_wp = self._get_const("theta_wp")
        Zr = self._get_const("Zr")
        Kc = self._get_const("Kc")
        p_tab = self._get_const("p_tab")
        u_max = self._get_const("u_max")

        # =========================================================
        # 2. TAW и Дефицит влаги
        # =========================================================
        TAW = 1000 * (theta_fc - theta_wp) * Zr
        Dr = 1000 * (theta_fc - theta_avg) * Zr
        Dr_star = np.clip(Dr, 0, TAW)

        # =========================================================
        # 3. Подготовка данных для ET0
        # =========================================================
        date = pd.DatetimeIndex(["2026-07-15"])
        df = pd.DataFrame(index=date)

        df["tmean"] = self.T_mean
        df["rh"] = self.RH_mean
        df["wind"] = self.wind_speed
        df["rs"] = self.solar_radiation_wm2 * 86400 / 1_000_000

        # =========================================================
        # 4. ET0
        # =========================================================
        ET0 = pyet.pm_fao56(
            tmean=df["tmean"],
            wind=df["wind"],
            rs=df["rs"],
            rh=df["rh"],
            pressure=self.pressure_hpa,
            elevation=self.elevation,
            lat=self.lat
        )
        ET0_value = float(ET0.iloc[0])

        # =========================================================
        # 5. ETc
        # =========================================================
        ETc = Kc * ET0_value

        # =========================================================
        # 6. Корректировка p_i
        # =========================================================
        p_i = np.clip(p_tab + 0.04 * (5 - ETc), 0.1, 0.8)

        # =========================================================
        # 7. RAW
        # =========================================================
        RAW = p_i * TAW

        # =========================================================
        # 8. Качество данных и осадки
        # =========================================================
        Q_i = 1
        R_i = 1 if self.rain_mm > 0 else 0

        # =========================================================
        # 9. Управляющее решение
        # =========================================================
        if (
                Q_i == 1 and
                Dr_star >= RAW and
                R_i == 0 and
                self.wind_speed <= u_max
        ):
            U_i = True
        else:
            U_i = False

        # =========================================================
        # Результат
        # =========================================================
        return U_i

    def get_decision(self) -> bool:
        """Публичный метод для получения результата"""
        result = self.__calculate()

        return result
