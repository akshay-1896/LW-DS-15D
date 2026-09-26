"""
=====================================================================
STUDENT RESULT MANAGEMENT SYSTEM
=====================================================================
A single beginner project that uses every topic taught so far:

Comments & Variables | Data Types | Strings & Type Conversion
Input & Output | Operators | Conditional Statements | Loops
Functions | List | Tuple | Set | Dictionary
Exception Handling | File Handling
=====================================================================
"""

# ---------------- Global Data (Variables & Data Structures) ----------------
students = []                                  # List -> one dictionary per student
subjects = ("Math", "Science", "English")      # Tuple -> fixed list of subjects
used_rolls = set()                             # Set -> keeps roll numbers unique
FILE_NAME = "students_data.txt"                # Variable holding the file name


# ============================== FUNCTIONS ==============================

def calculate_grade(average, passing_marks=40):
    """Decides the grade using conditional statements.
    'passing_marks' is a DEFAULT argument."""
    if average >= 90:
        grade = "A+"
    elif average >= 75:
        grade = "A"
    elif average >= 60:
        grade = "B"
    elif average >= passing_marks:
        grade = "C"
    else:
        grade = "F"
    return grade


def add_student():
    """Takes student details from the user, validates them, and stores them."""
    try:
        roll = int(input("Enter roll number: "))          # type conversion -> may raise ValueError

        if roll in used_rolls:                             # set membership check
            raise Exception("This roll number is already used!")   # raise keyword

        name = input("Enter student name: ")

        marks = {}          # empty dictionary to store subject-wise marks
        total = 0

        for subject in subjects:                            # for loop over a tuple
            mark = int(input(f"Enter marks in {subject} (0-100): "))
            if mark < 0 or mark > 100:                       # comparison + logical operator
                raise ValueError(f"Marks for {subject} must be between 0 and 100")
            marks[subject] = mark
            total += mark                                    # compound assignment operator

    except ValueError as ve:
        print("Invalid input:", ve)
        return
    except Exception as e:
        print("Error:", e)
        return
    else:
        average = total / len(subjects)                      # division operator
        grade = calculate_grade(average, passing_marks=35)    # keyword argument used here

        student = {
            "roll": roll,
            "name": name,
            "marks": marks,
            "total": total,
            "average": average,
            "grade": grade
        }

        students.append(student)      # list method
        used_rolls.add(roll)          # set method
        print(f"\nStudent {name} added successfully! Grade: {grade}\n")
    finally:
        print("-- Add student operation finished --\n")


def view_all_students():
    """Prints every student record using a for loop with range()."""
    if len(students) == 0:
        print("\nNo student records available yet.\n")
        return

    print("\n---------- All Student Records ----------")
    for i in range(len(students)):                # for loop with range() over a list
        s = students[i]                            # list indexing
        print(f"Roll No: {s['roll']} | Name: {s['name']} | "
              f"Total: {s['total']} | Average: {s['average']:.2f} | Grade: {s['grade']}")
    print("------------------------------------------\n")


def search_student(roll):
    """Searches for a student by roll number, using for-else."""
    for student in students:                       # for loop directly over a list
        if student["roll"] == roll:                 # comparison operator
            print(f"\nFound -> Name: {student['name']}, Grade: {student['grade']}")
            print("Marks:")
            for subject in student["marks"]:         # dictionary traversal (keys)
                print(f"   {subject}: {student['marks'][subject]}")
            break                                     # break statement
    else:
        print(f"\nNo student found with roll number {roll}.")


def delete_student(roll):
    """Deletes a student record by roll number, using list .remove()."""
    for student in students:                       # for loop to locate the matching record
        if student["roll"] == roll:                 # comparison operator
            students.remove(student)                 # list method -> removes this record
            used_rolls.remove(roll)                   # set method -> frees up the roll number
            print(f"\nStudent with roll number {roll} deleted successfully.\n")
            break                                      # break statement
    else:
        print(f"\nNo student found with roll number {roll}.\n")


def save_to_file():
    """Saves every record into a text file, 7 lines per student."""
    with open(FILE_NAME, "w") as file:               # 'w' mode + with statement
        for s in students:
            file.write(str(s["roll"]) + "\n")
            file.write(s["name"] + "\n")
            for subject in subjects:
                file.write(str(s["marks"][subject]) + "\n")
            file.write(str(s["average"]) + "\n")
            file.write(s["grade"] + "\n")
    print(f"\nAll records saved to '{FILE_NAME}' successfully.\n")


def load_from_file():
    """Loads records back from the file when the program starts."""
    try:
        with open(FILE_NAME, "r") as file:            # 'r' mode + with statement
            lines = file.readlines()                   # readlines()
    except FileNotFoundError:
        print(f"No saved file found. A new '{FILE_NAME}' will be created when you save.\n")
        return

    record_size = 7   # roll, name, 3 marks, average, grade = 7 lines per student
    for i in range(0, len(lines), record_size):        # range with a step value
        roll = int(lines[i][:-1])                       # slicing off the trailing "\n"
        name = lines[i + 1][:-1]

        marks = {}
        total = 0
        for j in range(len(subjects)):                  # nested for loop with range
            mark = int(lines[i + 2 + j][:-1])
            marks[subjects[j]] = mark
            total += mark

        average = float(lines[i + 5][:-1])
        grade = lines[i + 6][:-1]

        student = {
            "roll": roll, "name": name, "marks": marks,
            "total": total, "average": average, "grade": grade
        }
        students.append(student)
        used_rolls.add(roll)

    if students:
        print(f"Loaded {len(students)} student record(s) from '{FILE_NAME}'.\n")


def show_menu():
    print("========== STUDENT RESULT MANAGEMENT SYSTEM ==========")
    print("1. Add a new student")
    print("2. View all students")
    print("3. Search student by roll number")
    print("4. Delete a student")
    print("5. Save records to file")
    print("6. Exit")
    print("========================================================")


def main():
    load_from_file()          # try loading old data as soon as the program starts

    while True:                 # while loop, exits only using break
        show_menu()
        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            add_student()
        elif choice == "2":
            view_all_students()
        elif choice == "3":
            try:
                roll = int(input("Enter roll number to search: "))
                search_student(roll)
            except ValueError:
                print("\nPlease enter a valid numeric roll number.\n")
        elif choice == "4":
            try:
                roll = int(input("Enter roll number to delete: "))
                delete_student(roll)
            except ValueError:
                print("\nPlease enter a valid numeric roll number.\n")
        elif choice == "5":
            save_to_file()
        elif choice == "6":
            save_to_file()       # auto-save before exiting
            print("\nExiting program. All the best for results!")
            break                 # break statement
        else:
            print("\nInvalid choice, please enter a number between 1 and 6.\n")


if __name__ == "__main__":
    main()
