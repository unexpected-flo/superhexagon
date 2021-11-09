import pathlib as P

folder = P.Path("./train_data")
right = folder/"right"
left = folder/"left"
nothing = folder/"None"

right.mkdir(exist_ok=True)
left.mkdir(exist_ok=True)
nothing.mkdir(exist_ok=True)

train_files = folder.iterdir()
existing_right = len(list(right.iterdir()))
existing_left = len(list(left.iterdir()))
existing_nothing = len(list(nothing.iterdir()))

for file in train_files:
    if file.suffix == ".png":
        destination = file.stem.split(sep="_")[-1]
        if destination == "Key.right":
            file.rename(folder/"right"/"{}_{}.png".format(existing_right, destination))
            existing_right += 1
        if destination == "Key.left":
            file.rename(folder/"left"/"{}_{}.png".format(existing_left, destination))
            existing_left += 1
        if destination == "None":
            file.rename(folder/"None"/"{}_{}.png".format(existing_nothing, destination))
            existing_nothing += 1


