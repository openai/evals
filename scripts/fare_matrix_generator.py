import json
from itertools import permutations

template = """{"input": [{"role": "system", "content": "TASK:
Calculate the distance or fare from the fare matrix.
If distance is requested, output in the format \\"100.0 km\\"; if fare is requested, output in the format \\"10,000 yen\\".
Just output the result without outputting the reason.
If there is no corresponding route, just output \\"None.\\".

EXAMPLE1:
How much is the fare from Tokyo to Osaka?
+-------+--------+-------+
| 501.1 |  192.2 | Osaka |
+-------+--------+-------+
| 346.4 | Nagoya |  6880 |
+-------+--------+-------+
| Tokyo |  11500 | 14900 |
+-------+--------+-------+
distances(km) / fare(yen)
↓
14,900 yen

EXAMPLE2:
How far is the distance from Nagoya to Tokyo?
+-------+--------+-------+
| 501.1 |  192.2 | Osaka |
+-------+--------+-------+
| 346.4 | Nagoya |  6880 |
+-------+--------+-------+
| Tokyo |  11500 | 14900 |
+-------+--------+-------+
distances(km) / fare(yen)
↓
346.4 km

EXAMPLE3:
How much is the fare from Osaka to Nagoya?
+-------+--------+-------+
| 501.1 |  192.2 | Osaka |
+-------+--------+-------+
| 346.4 | Nagoya |  6880 |
+-------+--------+-------+
| Tokyo |  11500 | 14900 |
+-------+--------+-------+
distances(km) / fare(yen)
↓
6,880 yen
"}, {"role": "user", "content": "
""".replace(
    "\n", "\\n"
)

keihin_names = [
    "Hodogaya",
    "Hazawa",
    "Kohoku",
    "Tsuduki",
    "Keihin-Kawasaki",
    "Tamagawa",
]

keihin_distances = {
    "Hodogaya,Hazawa": "--",
    "Hodogaya,Kohoku": 5.6,
    "Hodogaya,Tsuduki": 8.6,
    "Hodogaya,Keihin-Kawasaki": 14.2,
    "Hodogaya,Tamagawa": 16.4,
    "Hazawa,Kohoku": 3.3,
    "Hazawa,Tsuduki": 6.3,
    "Hazawa,Keihin-Kawasaki": 11.9,
    "Hazawa,Tamagawa": 14.1,
    "Kohoku,Tsuduki": 3.0,
    "Kohoku,Keihin-Kawasaki": 8.6,
    "Kohoku,Tamagawa": 10.8,
    "Tsuduki,Keihin-Kawasaki": 5.6,
    "Tsuduki,Tamagawa": 7.8,
    "Keihin-Kawasaki,Tamagawa": 2.2,
}

keihin_fare = {
    "Hodogaya,Hazawa": "--",
    "Hodogaya,Kohoku": 110,
    "Hodogaya,Tsuduki": 160,
    "Hodogaya,Keihin-Kawasaki": 270,
    "Hodogaya,Tamagawa": 310,
    "Hazawa,Kohoku": 110,
    "Hazawa,Tsuduki": 160,
    "Hazawa,Keihin-Kawasaki": 270,
    "Hazawa,Tamagawa": 310,
    "Kohoku,Tsuduki": 60,
    "Kohoku,Keihin-Kawasaki": 160,
    "Kohoku,Tamagawa": 210,
    "Tsuduki,Keihin-Kawasaki": 110,
    "Tsuduki,Tamagawa": 160,
    "Keihin-Kawasaki,Tamagawa": 40,
}

yokosuka_names = [
    "Kariba",
    "Bessho",
    "Hino",
    "Konandai",
    "Kanazawa Nature Park",
    "Horiguchi Nokendai",
    "Namiki",
    "Aasahina",
    "Zushi",
    "Yokosuka",
    "Kinugasa",
    "Sahara",
    "Uraga",
    "MaboriKaigan",
]

yokosuka_distances = {
    "Kariba,Bessho": 3.2,
    "Kariba,Hino": 7.7,
    "Kariba,Konandai": 9.3,
    "Kariba,Kanazawa Nature Park": 13.0,
    "Kariba,Horiguchi Nokendai": 14.9,
    "Kariba,Namiki": 16.2,
    "Kariba,Aasahina": 13.5,
    "Kariba,Zushi": 19.1,
    "Kariba,Yokosuka": 21.3,
    "Kariba,Kinugasa": 26.6,
    "Kariba,Sahara": 27.8,
    "Kariba,Uraga": 31.7,
    "Kariba,MaboriKaigan": 32.8,
    "Bessho,Hino": 4.5,
    "Bessho,Konandai": 6.1,
    "Bessho,Kanazawa Nature Park": 9.8,
    "Bessho,Horiguchi Nokendai": 11.7,
    "Bessho,Namiki": 13.0,
    "Bessho,Aasahina": 10.3,
    "Bessho,Zushi": 15.9,
    "Bessho,Yokosuka": 18.1,
    "Bessho,Kinugasa": 23.4,
    "Bessho,Sahara": 24.6,
    "Bessho,Uraga": 28.5,
    "Bessho,MaboriKaigan": 29.6,
    "Hino,Konandai": 1.6,
    "Hino,Kanazawa Nature Park": 5.3,
    "Hino,Horiguchi Nokendai": 7.2,
    "Hino,Namiki": 8.5,
    "Hino,Aasahina": 5.8,
    "Hino,Zushi": 11.4,
    "Hino,Yokosuka": 13.6,
    "Hino,Kinugasa": 18.9,
    "Hino,Sahara": 20.1,
    "Hino,Uraga": 24.0,
    "Hino,MaboriKaigan": 25.1,
    "Konandai,Kanazawa Nature Park": 3.7,
    "Konandai,Horiguchi Nokendai": 5.6,
    "Konandai,Namiki": 6.9,
    "Konandai,Aasahina": 4.2,
    "Konandai,Zushi": 9.8,
    "Konandai,Yokosuka": 12.0,
    "Konandai,Kinugasa": 17.3,
    "Konandai,Sahara": 18.5,
    "Konandai,Uraga": 22.4,
    "Konandai,MaboriKaigan": 23.5,
    "Kanazawa Nature Park,Horiguchi Nokendai": "--",
    "Kanazawa Nature Park,Namiki": "--",
    "Kanazawa Nature Park,Aasahina": 3.1,
    "Kanazawa Nature Park,Zushi": 8.7,
    "Kanazawa Nature Park,Yokosuka": 10.9,
    "Kanazawa Nature Park,Kinugasa": 16.2,
    "Kanazawa Nature Park,Sahara": 17.4,
    "Kanazawa Nature Park,Uraga": 21.3,
    "Kanazawa Nature Park,MaboriKaigan": 22.4,
    "Horiguchi Nokendai,Namiki": "--",
    "Horiguchi Nokendai,Aasahina": 5.0,
    "Horiguchi Nokendai,Zushi": 10.6,
    "Horiguchi Nokendai,Yokosuka": 12.8,
    "Horiguchi Nokendai,Kinugasa": 18.1,
    "Horiguchi Nokendai,Sahara": 19.3,
    "Horiguchi Nokendai,Uraga": 23.2,
    "Horiguchi Nokendai,MaboriKaigan": 24.3,
    "Namiki,Aasahina": 6.3,
    "Namiki,Zushi": 11.9,
    "Namiki,Yokosuka": 14.1,
    "Namiki,Kinugasa": 19.4,
    "Namiki,Sahara": 20.6,
    "Namiki,Uraga": 24.5,
    "Namiki,MaboriKaigan": 25.6,
    "Aasahina,Zushi": 5.6,
    "Aasahina,Yokosuka": 7.8,
    "Aasahina,Kinugasa": 13.1,
    "Aasahina,Sahara": 14.9,
    "Aasahina,Uraga": 18.2,
    "Aasahina,MaboriKaigan": 19.3,
    "Zushi,Yokosuka": 2.2,
    "Zushi,Kinugasa": 7.5,
    "Zushi,Sahara": 8.7,
    "Zushi,Uraga": 12.6,
    "Zushi,MaboriKaigan": 13.7,
    "Yokosuka,Kinugasa": 5.3,
    "Yokosuka,Sahara": 6.5,
    "Yokosuka,Uraga": 10.4,
    "Yokosuka,MaboriKaigan": 11.5,
    "Kinugasa,Sahara": 1.2,
    "Kinugasa,Uraga": 5.1,
    "Kinugasa,MaboriKaigan": 6.2,
    "Sahara,Uraga": "--",
    "Sahara,MaboriKaigan": "--",
    "Uraga,MaboriKaigan": "--",
}

yokosuka_fare = {
    "Kariba,Bessho": 160,
    "Kariba,Hino": 310,
    "Kariba,Konandai": 360,
    "Kariba,Kanazawa Nature Park": 500,
    "Kariba,Horiguchi Nokendai": 590,
    "Kariba,Namiki": 590,
    "Kariba,Aasahina": 520,
    "Kariba,Zushi": 600,
    "Kariba,Yokosuka": 630,
    "Kariba,Kinugasa": 710,
    "Kariba,Sahara": 730,
    "Kariba,Uraga": 810,
    "Kariba,MaboriKaigan": 810,
    "Bessho,Hino": 210,
    "Bessho,Konandai": 260,
    "Bessho,Kanazawa Nature Park": 420,
    "Bessho,Horiguchi Nokendai": 500,
    "Bessho,Namiki": 500,
    "Bessho,Aasahina": 410,
    "Bessho,Zushi": 520,
    "Bessho,Yokosuka": 550,
    "Bessho,Kinugasa": 630,
    "Bessho,Sahara": 650,
    "Bessho,Uraga": 730,
    "Bessho,MaboriKaigan": 730,
    "Hino,Konandai": 100,
    "Hino,Kanazawa Nature Park": 260,
    "Hino,Horiguchi Nokendai": 390,
    "Hino,Namiki": 390,
    "Hino,Aasahina": 210,
    "Hino,Zushi": 400,
    "Hino,Yokosuka": 430,
    "Hino,Kinugasa": 510,
    "Hino,Sahara": 530,
    "Hino,Uraga": 610,
    "Hino,MaboriKaigan": 610,
    "Konandai,Kanazawa Nature Park": 210,
    "Konandai,Horiguchi Nokendai": 340,
    "Konandai,Namiki": 340,
    "Konandai,Aasahina": 160,
    "Konandai,Zushi": 360,
    "Konandai,Yokosuka": 390,
    "Konandai,Kinugasa": 470,
    "Konandai,Sahara": 490,
    "Konandai,Uraga": 570,
    "Konandai,MaboriKaigan": 570,
    "Kanazawa Nature Park,Horiguchi Nokendai": "--",
    "Kanazawa Nature Park,Namiki": "--",
    "Kanazawa Nature Park,Aasahina": 210,
    "Kanazawa Nature Park,Zushi": 330,
    "Kanazawa Nature Park,Yokosuka": 360,
    "Kanazawa Nature Park,Kinugasa": 440,
    "Kanazawa Nature Park,Sahara": 460,
    "Kanazawa Nature Park,Uraga": 540,
    "Kanazawa Nature Park,MaboriKaigan": 540,
    "Horiguchi Nokendai,Namiki": "--",
    "Horiguchi Nokendai,Aasahina": 330,
    "Horiguchi Nokendai,Zushi": 410,
    "Horiguchi Nokendai,Yokosuka": 450,
    "Horiguchi Nokendai,Kinugasa": 530,
    "Horiguchi Nokendai,Sahara": 550,
    "Horiguchi Nokendai,Uraga": 620,
    "Horiguchi Nokendai,MaboriKaigan": 620,
    "Namiki,Aasahina": 330,
    "Namiki,Zushi": 410,
    "Namiki,Yokosuka": 450,
    "Namiki,Kinugasa": 530,
    "Namiki,Sahara": 550,
    "Namiki,Uraga": 620,
    "Namiki,MaboriKaigan": 620,
    "Aasahina,Zushi": 210,
    "Aasahina,Yokosuka": 280,
    "Aasahina,Kinugasa": 360,
    "Aasahina,Sahara": 380,
    "Aasahina,Uraga": 460,
    "Aasahina,MaboriKaigan": 460,
    "Zushi,Yokosuka": 100,
    "Zushi,Kinugasa": 280,
    "Zushi,Sahara": 300,
    "Zushi,Uraga": 370,
    "Zushi,MaboriKaigan": 370,
    "Yokosuka,Kinugasa": 210,
    "Yokosuka,Sahara": 260,
    "Yokosuka,Uraga": 340,
    "Yokosuka,MaboriKaigan": 340,
    "Kinugasa,Sahara": 100,
    "Kinugasa,Uraga": 260,
    "Kinugasa,MaboriKaigan": 260,
    "Sahara,Uraga": "--",
    "Sahara,MaboriKaigan": "--",
    "Uraga,MaboriKaigan": "--",
}

not_exist_pair = [
    ("Tokyo", "Osaka"),
    ("Hiroshima", "Kokura"),
    ("Uraga", "Tokyo"),
    ("Sapporo", "Sendai"),
]


def generate_table(names, distances, fare):
    table = ""

    for name in reversed(names):
        table += "+"
        for i in range(len(names)):
            table += "-" * (len(names[i]) + 2) + "+"
        table += "\n"

        table += "|"
        for i in range(len(names)):
            if names[i] == name:
                table += f" {name} |"
            elif f"{names[i]},{name}" in distances:
                value = str(distances[f"{names[i]},{name}"])
                space_count = len(names[i]) - len(value) + 2 - 1
                table += f"{' ' * space_count}{value} |"
            else:
                value = str(fare[f"{name},{names[i]}"])
                space_count = len(names[i]) - len(value) + 2 - 1
                table += f"{' ' * space_count}{value} |"

        table += "\n"

    table += "+"
    for i in range(len(names)):
        table += "-" * (len(names[i]) + 2) + "+"
    table += "\n"

    note = "distances(km) / fare(yen)"

    table += " " * (sum(len(name) + 3 for name in names) - len(note)) + note

    return table


def format_expected(expected, unit, word):
    if expected == "--":
        return '["None."]'

    expected_text = "{expected:,} {unit}".format(expected=expected, unit=unit)

    return json.dumps([f"{expected_text}", f"The {word} from {a} to {b} is {expected_text}"])


def print_json_line(query, table, expected):
    print(
        template
        + query
        + table.replace("\n", "\\n")
        + '\\n↓\\n"    }  ],  "ideal": '
        + expected
        + "}"
    )


for names, distances, fare in [
    (keihin_names, keihin_distances, keihin_fare),
    (yokosuka_names, yokosuka_distances, yokosuka_fare),
]:
    table = generate_table(names, distances, fare)

    for a, b in permutations(names, 2):
        query = f"How much is the fare from {a} to {b}?\\n"
        expected = fare[f"{a},{b}"] if f"{a},{b}" in fare else fare[f"{b},{a}"]
        print_json_line(query, table, format_expected(expected, "yen", "fare"))

    for a, b in permutations(names, 2):
        query = f"How far is the distance from {a} to {b}?\\n"
        expected = distances[f"{a},{b}"] if f"{a},{b}" in distances else distances[f"{b},{a}"]
        print_json_line(query, table, format_expected(expected, "km", "distance"))

    for a, b in not_exist_pair:
        query = f"How much is the fare from {a} to {b}?\\n"
        print_json_line(query, table, format_expected("--", "yen", "fare"))

        query = f"How far is the distance from {a} to {b}?\\n"
        print_json_line(query, table, format_expected("--", "km", "distance"))
