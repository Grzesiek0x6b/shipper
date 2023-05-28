from abc import ABC, abstractmethod
from bisect import insort_left
from collections import defaultdict
from functools import partial
from itertools import chain, combinations
import math
import multiprocessing
import os
from queue import Empty
from random import random

import pygame
from mst import Graph
import solutions

import ui


planet_colors = {
    "Desert": ((168, 88, 2), (255, 132, 0)),
    "Fire": ((171, 2, 36), (255, 0, 51)),
    "Water": ((0, 32, 74), (0, 83, 191)),
    "Terran": ((0, 66, 22), (0, 191, 64)),
    "Gas": ((45, 1, 74), (115, 0, 191)),
    "IceA": ((0, 58, 61), (0, 181, 191)),
    "IceB": ((0, 58, 61), (0, 181, 191))
}



class SpaceObject(ABC):

    packages: int
    capacity: int
    targets: int
    storage: bool = False
    likely_assign: float = 1

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def active_color(self):
        pass
    
    @property
    @abstractmethod
    def full_name(self):
        pass

    @property
    @abstractmethod
    def produces(self):
        pass

    def get_sector(self, sectors):
        for sector in sectors:
            if sector.sector_content and self in sector.sector_content:
                return sector


class Planet(SpaceObject):
    type: str
    tier: int

    def __init__(self, type, tier, packages, capacity, targets):
        self.type = type
        self.tier = tier
        self.packages = packages
        self.capacity = capacity
        self.targets = targets
        self.likely_assign = (capacity/targets/100)**targets if targets > 1 else 1
        self.colonized = True

    @property
    def color(self):
        return planet_colors[self.type][0]

    @property
    def active_color(self):
        return planet_colors[self.type][1]
    
    @property
    def full_name(self):
        return self.full_type
    
    @property
    def full_type(self):
        return self.type + str(self.tier)
    
    @property
    def produces(self):
        return self.packages if self.colonized and not self.storage else 0


class TradeStation(SpaceObject):

    number: int

    def __init__(self, number):
        self.number = number
        self.packages = 12
        self.capacity = 36
        self.targets = 1
        self.likely_assign = 0.5
    
    @property
    def color(self):
        return (100, 125, 114)
    
    @property
    def active_color(self):
        return (177, 222, 202)
    
    @property
    def full_name(self):
        return "TS " + str(self.number)
    
    @property
    def produces(self):
        return self.packages


planets = [
    Planet("Desert", 1, 30, 60, 3),
    Planet("Fire", 1, 30, 60, 2),
    Planet("Water", 1, 36, 72, 2),
    Planet("Terran", 1, 18, 36, 1),
    Planet("Gas", 2, 22, 44, 1),
    Planet("Terran", 3, 27, 54, 1),
    Planet("Fire", 3, 24, 48, 1),
    Planet("Water", 3, 24, 48, 1),
    Planet("Gas", 4, 40, 80, 1),
    Planet("Desert", 3, 24, 48, 1),
    Planet("Fire", 4, 36, 72, 1),
    Planet("Desert", 4, 36, 72, 1),
    Planet("Water", 4, 38, 76, 1),
    Planet("Terran", 4, 35, 70, 1),
    Planet("IceA", 4, 40, 80, 1),
    Planet("IceB", 4, 50, 100, 1)
]

class Sector(ui.HexButton):

    def __init__(self, *args, location=(0,0),  **kwargs):
        super().__init__(*args, **kwargs)
        self.sector_location = location
        self.mouse_button_down = self.toogle
        self.sector_type = "hidden"
        self.sector_content = []

    def __eq__(self, other):
        return isinstance(other, Sector) and self.sector_location == other.sector_location
    
    def __hash__(self):
        return hash(self.sector_location)

    def __repr__(self):
        return f"Sector({self.tooltip})"

    @property
    def tooltip(self):
        return "\n".join([self._tooltip_header, self._tooltip_body])

    @tooltip.setter
    def tooltip(self, text):
        pass
    
    @property
    def _tooltip_header(self):
        return repr(self.sector_location) + " " + self.sector_types[self.sector_type].__doc__

    @property
    def _tooltip_body(self):
        if self.sector_type == "planets":
            return " & ".join(space_object.full_name for space_object in self.sector_content)
        elif self.sector_type == "station":
            return "Trade station"
        else:
            return ""

    @property
    def sector_type(self):
        return getattr(self, "_sector_type", None)
    
    @sector_type.setter
    def sector_type(self, name):
        
        @Sector._register_type
        def star():
            """Yellow star"""
            self.is_enabled = False
            self.unregister("mouse_button_down", all=True)
            self.background = (255,255,0)
            self.border = (255,255,255)
        
        @Sector._register_type
        def hidden():
            """Hidden sector"""
            self.background = (15,15,15)
            self.border = (40,40,40)

        @Sector._register_type
        def empty():
            """Empty sector"""
            self.background = (40,40,40)
            self.border = (60,60,60)
        
        @Sector._register_type
        def planets():
            """Occupied sector"""
            objects_number = len(self.sector_content)
            self.background = ui.colors_avg(obj.active_color for obj in self.sector_content)

        
        @Sector._register_type
        def station():
            """Occupied sector"""
            self.background = (193, 212, 210)

        if name in self.sector_types:
            self._sector_type = name
            self.text = self._tooltip_body
            self.sector_types[name]()

    @property
    def toogle(self):
        def handler(event):
            for func in set(getattr(self, "_toogle", [])):
                func(self)
        return handler
    
    @toogle.setter
    def toogle(self, func):
        if not hasattr(self, "_toogle"):
            self._toogle = set()
        self._toogle.add(func)

    @classmethod
    def _register_type(cls, func):
        if not hasattr(cls, "sector_types"):
            cls.sector_types = {}
        cls.sector_types[func.__name__] = func
    
    def distance(self, other):
        return math.dist(other.center, self.center)
    
    def is_adjacent(self, other):
        return math.isclose(self.distance(other), 2 * self.minimal_radius, rel_tol=0.05)
    
    def alias(self, targets):
        secalias = solutions.SectorAlias()
        secalias.location=self.sector_location
        secalias.type=self.sector_type
        secalias.content=[targets.index(so) for so in self.sector_content if isinstance(so, TradeStation) or so.colonized]
        secalias.targets=sum(so.targets for so in self.sector_content if isinstance(so, TradeStation) or so.colonized)
        secalias.center=self.center
        secalias.minimal_radius=self.minimal_radius
        return secalias


class App:
    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.sector_radius = 1
        self.uimanager = ui.Manager(screen_size=self.screen_size)
        self.computing = None
        self.warps_count = 0
        self.warps_lines = []
        self.assigment_lines = defaultdict(list)
        self.stations_count = 0
        self.stations = []
        self.mode = "reveal"
        for element in chain(self.init_sectors(), self.init_menu()):
            self.uimanager.add(element)

    def create_sector(self, left, top, radius, location):
        sector = Sector(left=left, top=top, radius=radius, location=location)
        sector.toogle = self.toogle_sector
        return sector

    def init_sectors(self):
        half_height = self.screen_size[1]/2
        self.sector_radius = half_height/(7*math.cos(math.radians(30)))

        leftmost_sector = self.create_sector(-self.sector_radius, 0, self.sector_radius, (0,0))

        sectors = [leftmost_sector]
        for x in range(7):
            if x:
                left, top = leftmost_sector.vertices[2]
                leftmost_sector = self.create_sector(left, top, self.sector_radius, (0, x))
                sectors.append(leftmost_sector)

            sector = leftmost_sector
            for i in range(7):
                if i % 2 == 1:
                    position = (sector.left + sector.radius * 3 / 2, sector.top - sector.minimal_radius)
                else:
                    position = (sector.left + sector.radius * 3 / 2, sector.top + sector.minimal_radius)
                sector = self.create_sector(*position, self.sector_radius, (i, x))
                sectors.append(sector)
        center_sec = min(sectors, key=lambda sec: math.dist(sec.center, (half_height,half_height)))
        center_sec.sector_type = "star"
        self.sectors = [sec for sec in sectors if math.dist(center_sec.center, sec.center)<6*self.sector_radius]
        return self.sectors

    def init_menu(self):
        stack = ui.StackPanel(left=self.screen_size[1] + 10, top=5, spacing=10)
        
        header = ui.Label(text="Checklist", width=200, height=30)

        btn1 = ui.ToogleButton(
            relative_left=0, width=200, height=30,
            text = "Sectors uncovered",
            background=(70,100,255), active_background=(70,255,100), border=None
        )
        btn2 = ui.ToogleButton(
            relative_left=0, width=200, height=30,
            text = "Planets placed",
            background=(15,15,15), active_background=(70,255,100), border=None
        )
        btn2.is_enabled = False
        btn3 = ui.ToogleButton(
            relative_left=0, width=200, height=30,
            text = "System configured",
            background=(15,15,15), active_background=(70,255,100), border=None
        )
        btn3.is_enabled = False
        btn4 = ui.ToogleButton(
            relative_left=0, width=200, height=30,
            text = "Solutions evaluated",
            background=(15,15,15), active_background=(70,255,100), border=None
        )
        btn4.is_enabled = False

        planets_list = ui.StackPanel(relative_top=20, spacing=2)
        planets_list.is_visible = False
        for btn in self.init_planets_list():
            planets_list.add(btn)
        
        self.planets_panels = ui.StackPanel(relative_top=20, spacing=20)
        self.planets_panels.is_visible = False
        for panel in self.init_planets_config():
            self.planets_panels.add(panel)
        
        results_panel = ui.StackPanel(relative_top=20, spacing=10)
        results_panel.is_visible = False
        for el in self.init_results_panel(btn4):
            results_panel.add(el)

        def uncovered(event=None):
            btn1.is_enabled = False
            btn2.is_enabled = True
            btn2.background = (70,100,255)
            planets_list.is_visible = True
            self.mode = "place-planets"
        
        def placed(event=None):
            btn2.is_enabled = False
            btn3.is_enabled = True
            btn3.background = (70,100,255)
            planets_list.is_visible = False
            self.planets_panels.is_visible = True
            self.mode = "configure-system"
        
        def configured(event=None):
            btn3.is_enabled = False
            btn4.background = (70,100,255)
            self.planets_panels.is_visible = False
            for sector in self.sectors:
                if sector.sector_type == "planets":
                    if not any(planet.colonized for planet in sector.sector_content):
                        sector.background = ui.colors_avg(obj.color for obj in sector.sector_content)
            results_panel.is_visible = True
            self.stations = [TradeStation(i+1) for i in range(self.stations_count)]
            self.computing = solutions.Compute(self)
            self.mode = "results-computing"

        btn1.changed = uncovered
        btn2.changed = placed
        btn3.changed = configured

        stack.add(header)
        stack.add(btn1)
        stack.add(btn2)
        stack.add(btn3)
        stack.add(btn4)
        stack.add(planets_list)
        stack.add(self.planets_panels)
        stack.add(results_panel)

        return (stack,)

    def init_planets_list(self):
        self._selected = None

        def create_button(*planets):
            if len(planets) == 1:
                planet = planets[0]
                return ui.ToogleButton(text=planet.full_type, tag=tuple(planets), width=200, height=30, background=planet.color, active_background=planet.active_color, display_checkmark=False)
            else:
                text = " & ".join(planet.full_type for planet in planets)
                bg = ui.colors_avg(planet.color for planet in planets)
                bga = ui.colors_avg(planet.active_color for planet in planets)
                return ui.ToogleButton(text=text, tag=tuple(planets), width=200, height=30, background=bg, active_background=bga, display_checkmark=False)

        buttons = [create_button(planets[0], planets[1])] + [create_button(planet) for planet in planets[2:]]
        
        def untoogle_others(event, btn):
            if btn.is_toogled:
                for button in buttons:
                    if button != btn and button.is_enabled and button.is_toogled:
                        button.is_toogled = False

        def set_selected(event):
            tags = [btn.tag for btn in buttons if btn.is_toogled]
            self._selected = tags[0] if tags else None

        for button in buttons:
            button.changed = partial(untoogle_others, btn=button)
            button.changed = set_selected
        
        return buttons

    def init_planets_config(self):

        warps_panel = ui.StackPanel(spacing=2, tag=None, vertical=False)
        warps_label = ui.Label(text="Warps:", width=50, height=30)
        warps_count = ui.Label(text="0", width=50, height=30)

        def warps_decrease(event):
            if self.warps_count > 0:
                self.warps_count -= 1
                warps_count.text = str(self.warps_count)

        def warps_increase(event):
            if self.warps_count < 12:
                self.warps_count += 1
                warps_count.text = str(self.warps_count)

        warps_decr = ui.Button(text="▼", width=48, height=30)
        warps_decr.mouse_button_down = warps_decrease
        warps_incr = ui.Button(text="▲", width=48, height=30)
        warps_incr.mouse_button_down = warps_increase
        warps_panel.add(warps_label)
        warps_panel.add(warps_count)
        warps_panel.add(warps_decr)
        warps_panel.add(warps_incr)

        stations_panel = ui.StackPanel(spacing=2, tag=None, vertical=False)
        stations_label = ui.Label(text="Stations:", width=50, height=30)
        stations_count = ui.Label(text="0", width=50, height=30)

        def stations_decrease(event):
            if self.stations_count > 0:
                self.stations_count -= 1
                stations_count.text = str(self.stations_count)

        def stations_increase(event):
            if self.stations_count < 3:
                self.stations_count += 1
                stations_count.text = str(self.stations_count)

        stations_decr = ui.Button(text="▼", width=48, height=30)
        stations_decr.mouse_button_down = stations_decrease
        stations_incr = ui.Button(text="▲", width=48, height=30)
        stations_incr.mouse_button_down = stations_increase
        stations_panel.add(stations_label)
        stations_panel.add(stations_count)
        stations_panel.add(stations_decr)
        stations_panel.add(stations_incr)


        panels = [warps_panel, stations_panel]
        
        def mark_colonized(event, btn, planet):
            planet.colonized = btn.is_toogled

        def mark_storage(event, btn, planet):
            planet.storage = btn.is_toogled

        for planet in planets:
            panel = ui.StackPanel(spacing=2, tag=planet)
            panel.is_visible = False
            panel.add(ui.Label(text=planet.full_type, width=200, height=30))
            colonized = ui.ToogleButton(text="colonized", border=None, width=200, height=30)
            colonized.is_toogled = True
            storage = ui.ToogleButton(text="storage", border=None, width=200, height=30)
            colonized.changed = partial(mark_colonized, btn=colonized, planet=planet)
            storage.changed = partial(mark_storage, btn=storage, planet=planet)
            panel.add(colonized)
            panel.add(storage)
            panels.append(panel)
        
        return panels
    
    def init_results_panel(self, menu_btn):
        orig_menu_btn_update = menu_btn.update

        def update_total_progress(bar, max_width):
            if self.computing:
                bar.width = min(self.computing.progress() * max_width, max_width)

        def update_consumer_progress(bar, i, max_width):
            if self.computing:
                ps = self.computing.subprogresses()
                if len(ps) > i:
                    p = ps[i]
                    bar.width = min((p[0]/p[1]) * max_width, max_width) if p[1] else 0

        progressbar = ui.StackPanel(spacing=0, relative_top=-20)
        total = ui.Background(color=(70,255,100), height=5)
        total.update = partial(update_total_progress, bar=total, max_width=200)
        progressbar.add(total)
        for i in range(4):
            bar = ui.Background(color=(70,100,255), height=5)
            bar.update = partial(update_consumer_progress, bar=bar, i=i, max_width=200)
            progressbar.add(bar)

        scores_label = ui.Label(text="Top 10 solutions:", width=200, height=30)
        scores_panel = ui.StackPanel(relative_top=0, spacing=5)

        show_all_arrows_btn = ui.Button(text="Show all directions", width=200, height=30, background=(0, 153, 102))
        show_all_arrows_btn.is_enabled = False

        sorted_buttons = []

        def clear_solution():
            for sector in self.sectors:
                if sector.sector_type == "station":
                    sector.sector_content = []
                    sector.sector_type = "empty"
            for line in self.warps_lines:
                self.uimanager.remove(line)
            self.warps_lines = []
            for line in chain.from_iterable(self.assigment_lines.values()):
                self.uimanager.remove(line)
            self.assigment_lines = defaultdict(list)

        def untoogle_other(btn):
            for _, _, _, button in sorted_buttons:
                if button != btn and button.is_enabled and button.is_toogled:
                    button.is_toogled = False
        
        def display_solution(solution):
            for sid, ts in zip(solution.ts_sectors, self.stations):
                sector = self.computing.sectors[sid]
                sector.sector_content = [ts]
                sector.sector_type = "station"
            g = Graph(len(solution.warps))
            for secid1, secid2 in combinations(range(len(solution.warps)), 2):
                g.add_edge(secid1, secid2, self.computing.sectors[solution.warps[secid1]].distance(self.computing.sectors[solution.warps[secid2]]))
            for secid1, secid2 in g.KruskalAlgo():
                (x1, y1), (x2, y2) = self.computing.sectors[solution.warps[secid1]].center, self.computing.sectors[solution.warps[secid2]].center
                y1 += 2/3 * self.sector_radius - 10
                y2 += 2/3 * self.sector_radius - 10
                x1 -= 10
                x2 -= 10
                line = ui.LineWithDots(x1=x1, y1=y1, x2=x2, y2=y2, color=(255,255,255))
                self.warps_lines.append(line)
                self.uimanager.add(line)
            for i in solution.directions:
                sourcesec = self.computing.targets[i].get_sector(self.computing.sectors)
                for j in solution.directions[i]:
                    destsec = self.computing.targets[j].get_sector(self.computing.sectors)
                    (x1, y1), (x2, y2) = sourcesec.center, destsec.center
                    y1 -= 1/3 * self.sector_radius - 5
                    y2 -= 1/3 * self.sector_radius - 5
                    x1 += (20 if self.computing.targets[i].full_name == "Fire1" else 0) + random() * 20 - 10
                    x2 += (20 if self.computing.targets[j].full_name == "Fire1" else 0) + random() * 20 - 10
                    line = ui.Arrow(x1=x1, y1=y1, x2=x2, y2=y2, color=ui.highlight(self.computing.targets[j].color, 2))
                    line.is_visible = False
                    line.tooltip=f"{self.computing.targets[i].full_name} → {self.computing.targets[j].full_name}"
                    self.assigment_lines[sourcesec].append(line)
                    self.uimanager.add(line)

        def toogle_button(event, btn, solution):
            clear_solution()
            if btn.is_toogled:
                untoogle_other(btn)
                display_solution(solution)
                show_all_arrows_btn.is_enabled = True

        def show_all_arrows(event):
            for lines in self.assigment_lines.values():
                for line in lines:
                    line.is_visible = True

        show_all_arrows_btn.mouse_button_down = show_all_arrows
        
        self.solution_number = 0


        def menu_btn_update():
            global solution_number
            orig_menu_btn_update()
            if self.computing is None:
                return
            for _ in range(10):
                try:
                    solution = self.computing.solutions.get_nowait()
                    self.solution_number += 1
                    button = ui.ToogleButton(
                        text=f"#{self.solution_number}",
                        width=200, height=30, display_checkmark=False,
                        foreground=(0,0,0), background=(198,188,83), active_background=(230,210,0))
                    button.highlight_tick = 30
                    button.changed = partial(toogle_button, btn=button, solution=solution)
                    insort_left(sorted_buttons, (solution.score, random(), self.solution_number, button))
                    del sorted_buttons[:-10]
                    scores_panel.clear()
                    scores_panel.add_range(reversed([sb[3] for sb in sorted_buttons]))
                except Empty:
                    if self.computing.task and not self.computing.task.is_alive():
                        menu_btn.update = orig_menu_btn_update
                        menu_btn.is_toogled = True
                        progressbar.is_visible = False
                        self.mode = "results"
                    break

        menu_btn.update = menu_btn_update

        return (progressbar, show_all_arrows_btn, scores_label, scores_panel)

    def toogle_sector(self, sector):
        if self.mode == "reveal":
            sector.sector_type = "empty" if sector.sector_type == "hidden" else "hidden"
        elif self.mode == "place-planets" and self._selected is not None:
            if sector.sector_type == "empty" and not any(sec.sector_content == self._selected for sec in self.sectors if sec.sector_content):
                sector.sector_content = self._selected
                sector.sector_type = "planets"
            elif sector.sector_type == "planets" and sector.sector_content == self._selected:
                sector.sector_content = []
                sector.sector_type = "empty"
        elif self.mode == "configure-system":
            for panel in self.planets_panels.elements:
                if panel.tag is not None:
                    panel.is_visible = False
            if sector.sector_type == "planets":
                for panel in self.planets_panels.elements:
                    if panel.tag in sector.sector_content:
                        panel.is_visible = True
        elif self.mode in ("results", "results-computing"):
            for line in chain.from_iterable(self.assigment_lines.values()):
                line.is_visible = False
            for line in self.assigment_lines[sector]:
                line.is_highlighted = True
                line.is_visible = True


def main():
    multiprocessing.freeze_support()
    pygame.init()
    pygame.display.set_caption('HS Shipper')
    screen_size = (1024, 768)
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    uimanager = App(screen_size).uimanager
    terminated = False
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            else:
                uimanager.handle(event)
        screen.fill((0, 0, 0))
        uimanager.update()
        uimanager.render(screen)
        pygame.display.flip()
        clock.tick(50)
    pygame.display.quit()


if __name__ == "__main__":
    main()
