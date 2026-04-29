from __future__ import annotations

from typing import Any, Callable

from InquirerPy import inquirer
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText


def ask_parsed(argument_name: str, parse: Callable[[str], Any]) -> Any:
    cached: dict[str, Any] = {}

    class V(Validator):
        def validate(self, document):
            text = document.text
            if text == "":
                raise ValidationError(message="Значение не может быть пустым", cursor_position=0)
            try:
                cached["value"] = parse(text)
            except Exception as ex:
                raise ValidationError(message=str(ex), cursor_position=len(text))

    inquirer.text(message=f'Введите "{argument_name}": ', validate=V()).execute()
    return cached["value"]


def print_error(text: str) -> None:
    print_formatted_text(FormattedText([("ansired", text)]))


def print_success(text: str) -> None:
    print_formatted_text(FormattedText([("ansigreen", text)]))


def pause(message: str = "Нажми Enter чтобы продолжить...") -> None:
    inquirer.text(message=message).execute()