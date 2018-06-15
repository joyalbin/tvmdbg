# coding: utf-8
# pylint: disable=too-many-arguments
"""Widgets for Curses-based CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tvm.tools.debug.cli import debugger_cli_common

RL = debugger_cli_common.RichLine


class NavigationHistoryItem(object):  # pylint: disable=too-few-public-methods
    """Individual item in navigation history."""

    def __init__(self, command, screen_output, scroll_position):
        """Constructor of NavigationHistoryItem.

        Args:
          command: (`str`) the command line text.
          screen_output: the screen output of the command.
          scroll_position: (`int`) scroll position in the screen output.
        """
        self.command = command
        self.screen_output = screen_output
        self.scroll_position = scroll_position


class CursesNavigationHistory(object):
    """Navigation history containing commands, outputs and scroll info."""

    BACK_ARROW_TEXT = "<--"
    FORWARD_ARROW_TEXT = "-->"
    EXIT_TEXT = "exit"
    HOME_TEXT = "home"
    HELP_TEXT = "help"

    def __init__(self, capacity):
        """Constructor of CursesNavigationHistory.

        Args:
          capacity: (`int`) How many items this object can hold. Each item consists
            of a command stirng, an output RichTextLines object and a scroll
            position.

        Raises:
          ValueError: If capacity is not a positive number.
        """
        if capacity <= 0:
            raise ValueError("In valid capacity value: %d" % capacity)

        self._capacity = capacity
        self._items = []
        self._pointer = -1

    def add_item(self, command, screen_output, scroll_position):
        """Add an item to the navigation histoyr.

        Args:
          command: command line text.
          screen_output: screen output produced for the command.
          scroll_position: (`int`) scroll position in the screen output.
        """
        if self._pointer + 1 < len(self._items):
            self._items = self._items[:self._pointer + 1]
        self._items.append(
            NavigationHistoryItem(command, screen_output, scroll_position))
        if len(self._items) > self._capacity:
            self._items = self._items[-self._capacity:]
        self._pointer = len(self._items) - 1

    def update_scroll_position(self, new_scroll_position):
        """Update the scroll position of the currently-pointed-to history item.

        Args:
          new_scroll_position: (`int`) new scroll-position value.

        Raises:
          ValueError: If the history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")
        self._items[self._pointer].scroll_position = new_scroll_position

    def size(self):
        """Get the size of widget items from list.

        Returns:
          ('int') length of widget items list.
        """
        return len(self._items)

    def pointer(self):
        """Get curses widget pointer.

        Returns:
          The pointer value.
        """
        return self._pointer

    def go_back(self):
        """Go back one place in the history, if possible.

        Decrease the pointer value by 1, if possible. Otherwise, the pointer value
        will be unchanged.

        Returns:
          The updated pointer value.

        Raises:
          ValueError: If history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")

        if self.can_go_back():
            self._pointer -= 1
        return self._items[self._pointer]

    def go_forward(self):
        """Go forward one place in the history, if possible.

        Increase the pointer value by 1, if possible. Otherwise, the pointer value
        will be unchanged.

        Returns:
          The updated pointer value.

        Raises:
          ValueError: If history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")

        if self.can_go_forward():
            self._pointer += 1
        return self._items[self._pointer]

    def can_go_back(self):
        """Test whether client can go back one place.

        Returns:
          (`bool`) Whether going back one place is possible.
        """
        return self._pointer >= 1

    def can_go_forward(self):
        """Test whether client can go forward one place.

        Returns:
          (`bool`) Whether going back one place is possible.
        """
        return self._pointer + 1 < len(self._items)

    def can_go_home(self):
        """Test whether client can go home place.

        Returns:
          (`bool`) Whether going back home place is possible.
        """
        if (self._pointer >= 0):
            if self._items[self._pointer].command == "home":
                return False
            return True
        else:
            return False

    def can_go_help(self):
        """Test whether client can go help place.

        Returns:
          (`bool`) Whether going back help place is possible.
        """
        if (self._pointer >= 0):
            if self._items[self._pointer].command == "help":
                return False
            return True
        else:
            return False

    def render(self,
               max_length,
               backward_command,
               forward_command,
               home_command,
               help_command,
               exit_command,
               latest_command_attribute="black_on_white",
               old_command_attribute="magenta_on_white"):
        """Render the rich text content of the single-line navigation bar.

        Args:
          max_length: (`int`) Maximum length of the navigation bar, in characters.
          backward_command: (`str`) command for going backward. Used to construct
            the shortcut menu item.
          forward_command: (`str`) command for going forward. Used to construct the
            shortcut menu item.
           latest_command_attribute: font attribute for lastest command.
           old_command_attribute: font attribute for old (non-latest) command.

        Returns:
          (`debugger_cli_common.RichTextLines`) the navigation bar text with
            attributes.

        """
        output = RL("| ")
        output += RL(
            self.BACK_ARROW_TEXT,
            (debugger_cli_common.MenuItem(None, backward_command)
             if self.can_go_back() else None))
        output += RL(" ")
        output += RL(
            self.FORWARD_ARROW_TEXT,
            (debugger_cli_common.MenuItem(None, forward_command)
             if self.can_go_forward() else None))

        output += RL(" | ")
        output += RL(self.HOME_TEXT,
            (debugger_cli_common.MenuItem(None, home_command)
             if self.can_go_home() else None))

        if self._items:
            command_attribute = (latest_command_attribute
                                 if (self._pointer == (len(self._items) - 1))
                                 else old_command_attribute)

            output_end = RL("| ")
            output_end += RL(self.HELP_TEXT,
                (debugger_cli_common.MenuItem(None, help_command)
                 if self.can_go_help() else None))
            output_end += RL(" | ")
            output_end += RL(self.EXIT_TEXT, debugger_cli_common.MenuItem(None, exit_command))
            output_end += RL(" |")

            output += RL(" | ")
            output_cmd = RL("")
            if self._pointer != len(self._items) - 1:
                output_cmd += RL("(-%d) " % (len(self._items) - 1 - self._pointer),
                             command_attribute)

            if (len(output)+len(output_cmd)+len(output_end)) < max_length:
                #maybe_truncated_command = self._items[self._pointer].command[
                #    :(max_length - (len(output)+len(output_end)+1))]
                #output_cmd += RL(maybe_truncated_command, command_attribute)

                space_need_size = max_length - (len(output)+len(output_cmd)+len(output_end) + 1)
                if space_need_size:
                    empty_line = " " * space_need_size
                    output_cmd += RL(empty_line)

        return debugger_cli_common.rich_text_lines_from_rich_line_list([output + output_cmd + output_end])
