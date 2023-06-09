from itertools import accumulate, chain, starmap
import math
import os
import random
import sys
import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Tuple, List


class Element(ABC):

    def __init__(self, **kwargs):
        self.left = kwargs.pop("left", 0)
        self.top = kwargs.pop("top", 0)
        self.width = kwargs.pop("width", 0)
        self.height = kwargs.pop("height", 0)
        self.relative_left = kwargs.pop("relative_left", None)
        self.relative_top = kwargs.pop("relative_top", None)
        self.tag = kwargs.pop("tag", None)
        self.tooltip = kwargs.pop("tooltip", None)
        self.font = None
        self.is_mouse_over = False
        self.is_enabled = True
        self.is_visible = True

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
    def left(self):
        return getattr(self, "_left", 0)

    @left.setter
    def left(self, value):
        changed = self.left != value
        self._left = value
        if changed:
            self.shape_changed(None)
    
    @property
    def top(self):
        return getattr(self, "_top", 0)

    @top.setter
    def top(self, value):
        changed = self.top != value
        self._top = value
        if changed:
            self.shape_changed(None)
    
    @property
    def width(self):
        return getattr(self, "_width", 0)

    @width.setter
    def width(self, value):
        changed = self.width != value
        self._width = value
        if changed:
            self.shape_changed(None)
    
    @property
    def height(self):
        return getattr(self, "_height", 0)

    @height.setter
    def height(self, value):
        changed = self.height != value
        self._height = value
        if changed:
            self.shape_changed(None)
    
    @property
    def shape_changed(self):
        def handler(event):
            for func in getattr(self, "_shape_changed", set()):
                func(event)
                # pass
        return handler
    
    @shape_changed.setter
    def shape_changed(self, func):
        if not hasattr(self, "_shape_changed"):
            self._shape_changed = set()
        self._shape_changed.add(func)
    
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


class Background(Element):
    
    def __init__(self, **kwargs):
        color = kwargs.pop("color", (0,0,0))
        super().__init__(**kwargs)
        self.color = color
    
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


class BoldLine(Element):

    def __init__(self, **kwargs):
        x1 = kwargs.pop("x1", 0)
        y1 = kwargs.pop("y1", 0)
        x2 = kwargs.pop("x2", 0)
        y2 = kwargs.pop("y2", 0)
        color = kwargs.pop("color", (0,0,0))
        is_highlighted = kwargs.pop("is_highlighted", False)
        thickness = kwargs.pop("thickness", 3)
        super().__init__(**kwargs)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.is_highlighted = is_highlighted
        self.thickness = thickness
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
        pygame.draw.line(screen, self.highlight_color, (self.x1, self.y1), (self.x2, self.y2), self.thickness)

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


class LineWithDots(BoldLine):

    def render(self, screen):
        super().render(screen)
        pygame.draw.circle(screen, self.highlight_color, (self.x1, self.y1), self.thickness * 1.5)
        pygame.draw.circle(screen, self.highlight_color, (self.x2, self.y2), self.thickness * 1.5)


class Arrow(BoldLine):

    def render(self, screen):
        super().render(screen)
        pygame.draw.polygon(screen, self.highlight_color, list(self.get_triangle_points()))

    def get_triangle_points(self):
        rotation = (math.atan2(self.y1 - self.y2, self.x1 - self.x2)) + math.pi
        triangle = [0, (3 * math.pi / 4), (5 * math.pi / 4)]
        for t in triangle:
            x = self.x2 + self.thickness * 2 * math.cos(t + rotation)
            y = self.y2 + self.thickness * 2 * math.sin(t + rotation)
            yield x, y


class Label(Element):

    def __init__(self, **kwargs):
        foreground = kwargs.pop("foreground", (255,255,255))
        background = kwargs.pop("background", None)
        text = kwargs.pop("text", "")
        super().__init__(**kwargs)
        self.foreground = foreground
        self.background = background
        self.text = text
    
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


class ButtonBase(Element):

    def __init__(self, **kwargs):
        foreground = kwargs.pop("foreground", (255,255,255))
        background = kwargs.pop("background", (0,0,0))
        border = kwargs.pop("border", (255,255,255))
        text = kwargs.pop("text", "")
        super().__init__(**kwargs)
        self.foreground = foreground
        self.background = background
        self.border = border
        self.text = text
        self.highlight_offset = 3
        self.max_highlight_ticks = 15
        self.highlight_tick = 0
        self.mouse_over = self._on_mouse_over
    
    def __post_init__(self):
        return super().__post_init__()
    
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


class Button(ButtonBase):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class ToogleButton(ButtonBase):
    
    def __init__(self, **kwargs):
        active_background = kwargs.pop("active_background", (0,0,0))
        display_checkmark = kwargs.pop("display_checkmark", True)
        super().__init__(**kwargs)
        self.active_background = active_background
        self.display_checkmark = display_checkmark
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


class HexButton(ButtonBase):

    def __init__(self, **kwargs):
        radius = kwargs.pop("radius", 0)
        super().__init__(**kwargs)
        self.radius = radius
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
        
    @property
    def radius(self):
        return getattr(self, "_radius", 0)

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.shape_changed(None)


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


class Panel(Element, EventHandler):

    def __init__(self, **kwargs):
        foreground = kwargs.pop("foreground", (255,255,255))
        background = kwargs.pop("background", (0,0,0))
        border = kwargs.pop("border", (255,255,255))
        super().__init__(**kwargs)
        self.foreground = foreground
        self.background = background
        self.border = border
        self.top_panel = self
        self.shape_changed = self.organize

    @property
    def elements(self):
        if not hasattr(self, "_elements"):
            self._elements = []
        return self._elements

    def add(self, element, redraw=True):
        element.font = self.font
        element.unregister("visibility_changed", func=self.top_panel.organize)
        element.visibility_changed = self.top_panel.organize
        element.unregister("shape_changed", func=self.top_panel.organize)
        element.shape_changed = self.top_panel.organize
        if isinstance(element, Panel):
            element.top_panel = self.top_panel
            element.unregister("shape_changed", func=element.organize)
        self.elements.append(element)
        if redraw:
            self.shape_changed(None)

    def add_range(self, elements, redraw=True):
        for element in elements:
            element.font = self.font
            element.unregister("visibility_changed", func=self.top_panel.organize)
            element.visibility_changed = self.top_panel.organize
            element.unregister("shape_changed", func=self.top_panel.organize)
            element.shape_changed = self.top_panel.organize
            if isinstance(element, Panel):
                element.top_panel = self.top_panel
                element.unregister("shape_changed", func=element.organize)
            self.elements.append(element)
        if redraw:
            self.shape_changed(None)

    def remove(self, element, redraw=True):
        self.elements.remove(element)
        if redraw:
            self.shape_changed(None)

    def clear(self, redraw=True):
        self.elements.clear()
        if redraw:
            self.shape_changed(None)
    
    def replace(self, elements, redraw=True):
        self.clear(False)
        self.add_range(elements, redraw)

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
    def organize(self, event=None):
        pass


class StackPanel(Panel):

    def __init__(self, **kwargs):
        spacing = kwargs.pop("spacing", 0)
        vertical = kwargs.pop("vertical", True)
        super().__init__(**kwargs)
        self.spacing = spacing
        self.vertical = vertical

    def organize(self, event=None):
        if len(self.elements) == 0:
            self.height = self.width = 0
            return

        def stack_vertically(top, element):
            if not element.is_visible:
                return top
            top += element.relative_top if element.relative_top else 0
            element._top = top
            element._left = self.left + (element.relative_left if element.relative_left else 0)
            if isinstance(element, Panel):
                element.organize(event)
            return top + element.height + self.spacing

        def stack_horizontally(left, element):
            if not element.is_visible:
                return left
            left += element.relative_left if element.relative_left else 0
            element._left = left
            element._top = self.top + (element.relative_top if element.relative_top else 0)
            if isinstance(element, Panel):
                element.organize(event)
            return left + element.width + self.spacing
             
        list(accumulate(
            self.elements,
            stack_vertically if self.vertical else stack_horizontally,
            initial=self.top if self.vertical else self.left))
        
        last = self.elements[-1] if self.elements else None
        if self.vertical:
            self._width = max(element.width for element in self.elements)
            self._height = last.top + last.height - self.top if last else 0
        else:
            self._width = last.left + last.width - self.left if last else 0
            self._height = max(element.height for element in self.elements)


class Manager(EventHandler):

    elements: List[Element]

    def __init__(self, *elements, screen_size):
        self.screen_size = screen_size
        self._elements = [Background(width=screen_size[0], height=screen_size[1])]
        pygame.font.init()
        self.font = pygame.Font(resource_path("seguisym.ttf"), 18)
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

    def _show_tooltip(self, event):
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


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)