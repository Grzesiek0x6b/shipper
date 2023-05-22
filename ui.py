from itertools import accumulate, chain, starmap
import math
import random
import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Tuple, List

@dataclass
class Element(ABC):

    left: float = 0
    top: float = 0
    relative_left: float = None
    relative_top: float = None
    tag: Any = None
    tooltip: Any = None
    _tooltip: Any = field(init=False, repr=False)

    @abstractmethod
    def __post_init__(self):
        self.font = None
        self.is_mouse_over = False
        self.is_enabled = True
        self.is_visible = True
        self._tooltip = None

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def render(self, screen):
        pass

    @abstractmethod
    def hittest(self, position):
        pass

    def unregister(self, eventname, func=None, all=False):
        if all:
            getattr(self, "_" + eventname, set()).clear()
        else:
            try:
                getattr(self, "_" + eventname, set()).remove(func)
            except KeyError:
                pass
    
    @property
    @abstractmethod
    def tooltip(self):
        return getattr(self, "_tooltip", None)

    @tooltip.setter
    @abstractmethod
    def tooltip(self, content):
        self._tooltip = content

    @property
    def is_enabled(self):
        return getattr(self, "_is_enabled", True)
    
    @is_enabled.setter
    def is_enabled(self, value):
        if self.is_enabled != value:
            self._is_enabled = value
            self.enabled_changed(None)

    @property
    def enabled_changed(self):
        def handler(event):
            for func in getattr(self, "_enabled_changed", set()):
                func(event)
        return handler
    
    @enabled_changed.setter
    def enabled_changed(self, func):
        if not hasattr(self, "_enabled_changed"):
            self._enabled_changed = set()
        self._enabled_changed.add(func)

    @property
    def is_visible(self):
        return getattr(self, "_is_visible", True)
    
    @is_visible.setter
    def is_visible(self, value):
        if self.is_visible != value:
            self._is_visible = value
            self.visibility_changed(None)

    @property
    def visibility_changed(self):
        def handler(event):
            for func in getattr(self, "_visibility_changed", set()):
                func(event)
        return handler
    
    @visibility_changed.setter
    def visibility_changed(self, func):
        if not hasattr(self, "_visibility_changed"):
            self._visibility_changed = set()
        self._visibility_changed.add(func)

    @property
    def mouse_over(self):
        def handler(event):
            self.is_mouse_over = True
            for func in set(getattr(self, "_mouse_over", [])):
                func(event)
        return handler
    
    @mouse_over.setter
    def mouse_over(self, func):
        if not hasattr(self, "_mouse_over"):
            self._mouse_over = set()
        self._mouse_over.add(func)

    @property
    def mouse_out(self):
        def handler(event):
            self.is_mouse_over = False
            for func in set(getattr(self, "_mouse_out", [])):
                func(event)
        return handler
    
    @mouse_out.setter
    def mouse_out(self, func):
        if not hasattr(self, "_mouse_out"):
            self._mouse_out = set()
        self._mouse_out.add(func)
    
    @property
    def mouse_button_down(self):
        def handler(event):
            for func in set(getattr(self, "_mouse_button_down", [])):
                func(event)
        return handler
    
    @mouse_button_down.setter
    def mouse_button_down(self, func):
        if not hasattr(self, "_mouse_button_down"):
            self._mouse_button_down = set()
        self._mouse_button_down.add(func)

    @property
    def mouse_button_up(self):
        def handler(event):
            for func in set(getattr(self, "_mouse_button_up", [])):
                func(event)
        return handler
    
    @mouse_button_up.setter
    def mouse_button_up(self, func):
        if not hasattr(self, "_mouse_button_up"):
            self._mouse_button_up = set()
        self._mouse_button_up.add(func)

    @property
    def mouse_wheel(self):
        def handler(event):
            for func in set(getattr(self, "_mouse_wheel", [])):
                func(event)
        return handler
    
    @mouse_wheel.setter
    def mouse_wheel(self, func):
        if not hasattr(self, "_mouse_wheel"):
            self._mouse_wheel = set()
        self._mouse_wheel.add(func)


@dataclass
class Background(Element):
    
    width: float = 0
    height: float = 0
    color: Tuple[int, ...] = (0,0,0)

    def __post_init__(self):
        super().__post_init__()
    
    def update(self):
        pass
    
    def render(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
    
    def hittest(self, position):
        return self.rect.collidepoint(position)

    @property
    def rect(self):
        return pygame.Rect(self.left, self.top, self.width, self.height)
    
    @property
    def tooltip(self):
        return super().tooltip
    
    @tooltip.setter
    def tooltip(self, text):
        super(__class__, self.__class__).tooltip.__set__(self, text)

@dataclass
class BoldLine(Element):

    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    color: Tuple[int, ...] = (0,0,0)
    is_highlighted: bool = False
    width: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.highlight_offset = 3
        self.max_highlight_ticks = 60
        self.highlight_tick = 0

    def update(self):
        if self.is_highlighted:
            if self.highlight_tick > 0:
                self.highlight_tick -= 1
            else:
                self.is_highlighted = False

    @property
    def is_highlighted(self):
        return getattr(self, "_is_highlighted", False)
    
    @is_highlighted.setter
    def is_highlighted(self, value):
        self._is_highlighted = value
        if value:
            self.highlight_tick = getattr(self, "max_highlight_ticks", 0)

    @property
    def highlight_color(self):
        offset = self.highlight_offset * self.highlight_tick
        return highlight(self.color, offset)

    def render(self, screen):
        pygame.draw.line(screen, self.highlight_color, (self.x1, self.y1), (self.x2, self.y2), self.width)

    def hittest(self, position):
        x, y = position
        if self.x1 == self.x2:
            return math.isclose(x, self.x1, rel_tol=0.01) and min(self.y1, self.y2) <= y <= max(self.y1, self.y2)
        elif self.y1 == self.y2:
            return math.isclose(y, self.y1, rel_tol=0.01) and min(self.x1, self.x2) <= x <= max(self.x1, self.x2)
        else:
            a = (self.y2 - self.y1)/(self.x2 - self.x1)
            b = self.y1 - a * self.x1
            return math.isclose(y, a * x + b, rel_tol=0.05) and min(self.x1, self.x2) <= x <= max(self.x1, self.x2) and min(self.y1, self.y2) <= y <= max(self.y1, self.y2)

    @property
    def tooltip(self):
        return super().tooltip
    
    @tooltip.setter
    def tooltip(self, text):
        super(__class__, self.__class__).tooltip.__set__(self, text)

@dataclass
class Label(Element):
    
    width: float = 0
    height: float = 0
    foreground: Tuple[int, ...] = (255,255,255)
    background: Tuple[int, ...] = None
    text: str = ""

    def __post_init__(self):
        super().__post_init__()
    
    def update(self):
        pass
    
    def render(self, screen):
        if self.text:
            label = self.font.render(self.text, True, self.foreground, self.background)
            left = self.left + (self.width - label.get_width()) / 2
            top = self.top + (self.height - label.get_height()) / 2
            screen.blit(label, (left, top))
    
    def hittest(self, position):
        return self.rect.collidepoint(position)

    @property
    def rect(self):
        return pygame.Rect(self.left, self.top, self.width, self.height)
    
    @property
    def tooltip(self):
        return super().tooltip
    
    @tooltip.setter
    def tooltip(self, text):
        super(__class__, self.__class__).tooltip.__set__(self, text)


class Tooltip(Label):

    def hittest(self, position):
        return False
     
    @property
    def tooltip(self):
        return None
    
    @tooltip.setter
    def tooltip(self, text):
        pass


def highlight(color, offset):
    brighten = lambda x, y: x + y if x + y < 255 else 255
    return tuple(brighten(x, offset) for x in color)


@dataclass
class ButtonBase(Element):

    foreground: Tuple[int, ...] = (255,255,255)
    background: Tuple[int, ...] = (0,0,0)
    border: Tuple[int, ...] = (255,255,255)
    text: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.highlight_offset = 3
        self.max_highlight_ticks = 15
        self.highlight_tick = 0
        self.mouse_over = self._on_mouse_over
    
    def update(self):
        if not self.is_mouse_over and self.highlight_tick > 0:
            self.highlight_tick -= 1

    @property
    def highlight_background(self):
        offset = self.highlight_offset * self.highlight_tick
        return highlight(self.background, offset)
    
    def _on_mouse_over(self, event):
        self.highlight_tick = self.max_highlight_ticks

    @property
    def tooltip(self):
        return super().tooltip
    
    @tooltip.setter
    def tooltip(self, text):
        super(__class__, self.__class__).tooltip.__set__(self, text)


@dataclass
class Button(ButtonBase):

    width: float = 0
    height: float = 0
    
    def __post_init__(self):
        super().__post_init__()

    def render(self, screen):
        pygame.draw.rect(screen, self.highlight_background, self.rect)
        if self.border:
            pygame.draw.rect(screen, self.border, self.rect, 3)
        if self.text:
            label = self.font.render(self.text, True, self.foreground)
            left = self.left + (self.width - label.get_width()) / 2
            top = self.top + (self.height - label.get_height()) / 2
            screen.blit(label, (left, top))
    
    def hittest(self, position):
        return self.rect.collidepoint(position)

    @property
    def rect(self):
        return pygame.Rect(self.left, self.top, self.width, self.height)


@dataclass
class ToogleButton(ButtonBase):

    width: float = 0
    height: float = 0
    active_background: Tuple[int, ...] = (0,0,0)
    display_checkmark: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self._is_toogled = False
        self.mouse_button_down = self._switch

    def render(self, screen):
        pygame.draw.rect(screen, self.active_background if self._is_toogled else self.highlight_background, self.rect)
        if self.border: 
            pygame.draw.rect(screen, self.border, self.rect, 1)
        if self.display_checkmark:
            checkmark = self.font.render("☑" if self._is_toogled else "☐", True, self.foreground)
            screen.blit(checkmark, (self.left + 2, self.top))
            checkmark_rect = checkmark.get_rect()
        else:
            checkmark_rect = pygame.Rect(0,0,0,0)
        if self.text:
            label = self.font.render(self.text, True, self.foreground)
            left = self.left + checkmark_rect.width + (self.width - checkmark_rect.width - label.get_width()) / 2
            top = self.top + (self.height - label.get_height()) / 2
            screen.blit(label, (left, top))
    
    def hittest(self, position):
        return self.rect.collidepoint(position)
    
    @property
    def changed(self):
        def handler(event):
            for func in getattr(self, "_changed", set()):
                func(event)
        return handler
    
    @changed.setter
    def changed(self, func):
        if not hasattr(self, "_changed"):
            self._changed = set()
        self._changed.add(func)

    def _switch(self, event):
        self.is_toogled = not self.is_toogled

    @property
    def is_toogled(self):
        return self._is_toogled
    
    @is_toogled.setter
    def is_toogled(self, value):
        if self._is_toogled != value:
            self._is_toogled = value
            self.changed(None)

    @property
    def rect(self):
        return pygame.Rect(self.left, self.top, self.width, self.height)


@dataclass
class HexButton(ButtonBase):

    radius: float = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.vertices = self.compute_vertices()

    def compute_vertices(self):
        half_radius = self.radius / 2
        minimal_radius = self.minimal_radius
        return [
            (self.left, self.top),
            (self.left - half_radius, self.top + minimal_radius),
            (self.left, self.top + 2 * minimal_radius),
            (self.left + self.radius, self.top + 2 * minimal_radius),
            (self.left + 3 * half_radius, self.top + minimal_radius),
            (self.left + self.radius, self.top),
        ]

    @property
    def minimal_radius(self) -> float:
        return self.radius * math.cos(math.radians(30))

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.radius / 2, self.top + self.minimal_radius)
    
    def render(self, screen):
        rect = pygame.draw.polygon(screen, self.highlight_background, self.vertices)
        pygame.draw.aalines(screen, self.border, closed=True, points=self.vertices)
        if self.text:
            label = self.font.render(self.text, True, self.foreground)
            left = rect.left + (rect.width - label.get_width()) / 2
            top = rect.top + (rect.height - label.get_height()) / 2
            screen.blit(label, (left, top))

    def hittest(self, position):
        return math.dist(position, self.center) < self.minimal_radius


class EventHandler:
    
    mouse_events = {
        pygame.MOUSEMOTION: attrgetter("mouse_over"),
        pygame.MOUSEBUTTONDOWN: attrgetter("mouse_button_down"),
        pygame.MOUSEBUTTONUP: attrgetter("mouse_button_up"),
        pygame.MOUSEWHEEL: attrgetter("mouse_wheel")
    }

    @property
    def handle(self):
        def handler(event):
            if event.type in self.mouse_events.keys():
                mouse_pos = pygame.mouse.get_pos()
                for element in self.elements:
                    if not element.is_visible:
                        continue
                    else:
                        if element.hittest(mouse_pos):
                            if element.is_enabled:
                                if isinstance(element, EventHandler):
                                    element.handle(event)
                                else:
                                    self.mouse_events[event.type](element)(event)
                        else:
                            if isinstance(element, EventHandler):
                                element.handle(event)
                            else:
                                element.mouse_out(event)
            for func in getattr(self, "_handle", set()):
                func(event)
        return handler
    
    @handle.setter
    def handle(self, func):
        if not hasattr(self, "_handle"):
            self._handle = set()
        self._handle.add(func)
    
    @property
    @abstractmethod
    def elements(self):
        pass


@dataclass
class Panel(Element, EventHandler):

    foreground: Tuple[int, ...] = (255,255,255)
    background: Tuple[int, ...] = (0,0,0)
    border: Tuple[int, ...] = (255,255,255)
    
    def __post_init__(self):
        super().__post_init__()

    @property
    def elements(self):
        if not hasattr(self, "_elements"):
            self._elements = []
        return self._elements

    def add(self, element):
        element.font = self.font
        element.visibility_changed = lambda _: self._organize()
        self.elements.append(element)
        self._organize()

    def add_range(self, elements):
        for element in elements:
            element.font = self.font
            element.visibility_changed = lambda _: self._organize()
            self.elements.append(element)
        self._organize()

    def remove(self, element):
        self.elements.remove(element)
        self._organize()

    def clear(self):
        self.elements.clear()
        self._organize()
    
    @property
    def font(self):
        return getattr(self, "_font", None)
    
    @font.setter
    def font(self, value):
        self._font = value
        for element in self.elements:
            element.font = value

    @property
    def tooltip(self):
        return super().tooltip
    
    @tooltip.setter
    def tooltip(self, text):
        super(__class__, self.__class__).tooltip.__set__(self, text)

    def update(self):
        for element in self.elements:
            element.update()
    
    def render(self, screen):
        for element in self.elements:
            if element.is_visible:
                element.render(screen)
    
    def hittest(self, position):
        for element in self.elements:
            if element.hittest(position):
                return True
        return False
    
    @abstractmethod
    def _organize(self):
        pass


@dataclass
class StackPanel(Panel):

    spacing: float = 0
    vertical: bool = True

    def _organize(self):
        if len(self.elements) == 0:
            self.height = self.width = 0
            return

        def stack_vertically(top, element):
            if not element.is_visible:
                return top
            top += element.relative_top if element.relative_top else 0
            element.top = top
            element.left = self.left + (element.relative_left if element.relative_left else 0)
            if isinstance(element, Panel):
                element._organize()
            return top + element.height + self.spacing

        def stack_horizontally(left, element):
            if not element.is_visible:
                return left
            left += element.relative_left if element.relative_left else 0
            element.left = left
            element.top = self.top + (element.relative_top if element.relative_top else 0)
            if isinstance(element, Panel):
                element._organize()
            return left + element.width + self.spacing
             
        list(accumulate(
            self.elements,
            stack_vertically if self.vertical else stack_horizontally,
            initial=self.top if self.vertical else self.left))
        
        last = self.elements[-1] if self.elements else None
        if self.vertical:
            self.width = max(element.width for element in self.elements)
            self.height = last.top + last.height - self.top if last else 0
        else:
            self.width = last.left + last.width - self.left if last else 0
            self.height = max(element.height for element in self.elements)



class Manager(EventHandler):

    elements: List[Element]

    def __init__(self, *elements, screen_size):
        self.screen_size = screen_size
        self._elements = [Background(width=screen_size[0], height=screen_size[1])]
        pygame.font.init()
        self.font = pygame.Font("seguisym.ttf", 18)
        for element in elements:
            self.add(element)
        self.handle = self._show_tooltip

    @property
    def elements(self):
        return chain(self._elements, (self.tooltip,))

    @property
    def tooltip(self):
        if not hasattr(self, "_tooltip"):
            self._tooltip = Tooltip(width=100, height=30, foreground=(33,33,30), background=(226,227,209))
            self._tooltip.is_visible = self._tooltip.is_enabled = False
            self._tooltip.font = self.font

        return self._tooltip
    
    @tooltip.setter
    def tooltip(self, args):
        text, position = args
        if text:
            self.tooltip.text = text
            self.tooltip.left, self.tooltip.top = position[0]+10, position[1]+10
            self.tooltip.is_visible = True
        else:
            self.tooltip.is_visible = False

    def add(self, element):
        element.font = self.font
        self._elements.append(element)
    
    def remove(self, element):
        self._elements.remove(element)

    def update(self):
        for element in self.elements:
            element.update()

    def render(self, screen):
        for element in self.elements:
            if element.is_visible:
                element.render(screen)

    def _show_tooltip(self, event: pygame.Event):
        if event.type in self.mouse_events.keys():
            mouse_pos = pygame.mouse.get_pos()
            for element in self.elements:
                if not element.is_visible:
                    continue
                else:
                    if element.hittest(mouse_pos):
                        self.tooltip = (element.tooltip, mouse_pos)


def get_random_color(min_=150, max_=255):
    return tuple(random.choices(list(range(min_, max_)), k=3))


def colors_avg(colors):
    colors = list(colors)
    count = len(colors)
    acc = (0,0,0)
    for color in colors:
        acc = (acc[0]+color[0], acc[1]+color[1], acc[2]+color[2])
    return acc[0]/count, acc[1]/count, acc[2]/count
