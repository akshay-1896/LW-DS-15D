# Student Result Management System — Topic Guide

This is a single console project, **`student_result_manager.py`**, built so it naturally uses every topic taught up to page 44 (Chapter 17 – File Handling) of the Python book. It's a menu-driven app where a teacher can add students, view them, search by roll number, delete a student, and save/load records from a file — so nothing feels forced in; each feature exists because the app genuinely needs it.

## How to run it

```
python student_result_manager.py
```

It shows a menu (Add / View / Search / Delete / Save / Exit) in a loop. Try adding 2–3 students, viewing them, searching by roll number, saving, and exiting — then run it again and watch it reload your saved data automatically.

## Where each topic is used

| Topic (Chapter) | Where it appears in the code |
|---|---|
| **Comments & Variables** | `#` comments throughout; `students`, `subjects`, `FILE_NAME`, `roll`, `name` etc. |
| **Data Types** | `int` (roll, marks), `float` (average), `str` (name), `bool` (comparison results), plus list/tuple/set/dict |
| **Strings & Type Conversion** | `int(input(...))`, `str(s["roll"])`, `float(lines[i+5][:-1])`, and slicing `line[:-1]` to strip the newline |
| **Input & Output** | `input()` for every user prompt; `print()` with f-strings like `f"Roll No: {s['roll']}..."` |
| **Operators** | `+=` (compound assignment), `/` (average), `==`, `<`, `>=` (comparisons), `or` (logical) |
| **Conditional Statements** | `if / elif / else` inside `calculate_grade()` and the main menu |
| **Loops** | `for` loop over a tuple (`subjects`), over a list (`students`), over `range(len(...))`, over `range(start, stop, step)` in `load_from_file()`, and a `while True` menu loop with `break` |
| **Loop control (break/continue/else)** | `break` to exit the menu and to stop `search_student()` once found; `for...else` in `search_student()` runs only if the student is *not* found |
| **Functions** | `calculate_grade()`, `add_student()`, `view_all_students()`, `search_student()`, `delete_student()`, `save_to_file()`, `load_from_file()`, `show_menu()`, `main()` — includes a **default argument** (`passing_marks=40`) and a **keyword argument call** (`calculate_grade(average, passing_marks=35)`) |
| **List** | `students = []`, `.append()`, indexing with `students[i]`, and `.remove()` in `delete_student()` |
| **Tuple** | `subjects = ("Math", "Science", "English")` — fixed and never changed |
| **Set** | `used_rolls = set()`, `.add()`, `roll in used_rolls` to block duplicate roll numbers, and `.remove()` in `delete_student()` to free a roll number back up |
| **Dictionary** | Each student is a `dict`; `marks` is a `dict`; traversal with `for subject in student["marks"]` |
| **Exception Handling** | `try / except / else / finally` in `add_student()`; a specific `except ValueError` before a generic `except Exception`; `raise` used twice (duplicate roll, invalid marks); `except FileNotFoundError` in `load_from_file()` |
| **File Handling** | `open()` with `'w'` and `'r'` modes, always through `with`; `.write()`, `.readlines()` |

## A few design notes (in case a student asks "why this way?")

- **Records are saved 7 lines per student** (roll, name, 3 marks, average, grade) instead of using `,`-joined lines. This means reading them back only needs `readlines()` + `range(start, stop, step)` + slicing — all things explicitly taught — with no need for `.split()`, which the book hasn't introduced yet.
- **`search_student()` uses `for...else`** specifically to demonstrate that the `else` on a loop only runs if `break` was never hit — a subtle point worth highlighting to the student.
- **Grades are intentionally simple** (`A+/A/B/C/F`) so the conditional-statement logic stays easy to trace by hand.

## Suggested extensions (once ready to stretch a bit further)

- Add a "Sort students by average" option — good motivation for sorting concepts later on.
- Track a class-wide **average of averages** using the existing `for` loop skills.
