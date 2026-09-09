import json, sys

def get_json_shape(data, indent=0) -> None:
    spacing=" " * indent

    if isinstance(data, dict):
        print(f"{spacing}Dict containing {len(data)} keys:")
        for key, value in data.items():
            print(f"{spacing}- {key}: ", end="")
            get_json_shape(value, indent + 1)

    elif isinstance(data, list):
        print(f"{spacing}List containing {len(data)} items")
        if len(data) > 0:
            print(f"{spacing} [Example of Item Structure]:")
            get_json_shape(data[0], indent + 2)

    else:
        print(f"{type(data).__name__}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: please provide a filename to check the shape.")
        print("Usage: python check_shape.py <filename.json>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        print(f"\n-- Structure of {filename} ---")
        get_json_shape(data)

    except FileNotFoundError:
        print(f"Error: the file {filename} was not found")
    except json.JSONDecodeError:
        print(f"Error: {filename} is not a valid json file")