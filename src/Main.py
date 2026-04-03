from ChordFunctions import *
from Timings import KeyTiming
from Rhythm import RhythmHMM
from Models import OrnamentNoteMCs, ChordHMM, BassGen, MusicGen
from SongInfo import TrainingDataProcessedInfo
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train commands
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", "-d", required=True, help="Path to training data.")
    train_parser.add_argument("--output", "-o", required=True, help="Path to store model")
    train_parser.add_argument(
        "--models",
        nargs="+",
        choices=["chordhmm", "bassgen", "rhythmhmm", "ornamentmcs"],
        required=True,
        help="One or more models to train"
    )

    # Generate commands
    gen_parser = subparsers.add_parser("generate", help="Generate music")
    gen_parser.add_argument("--model", required=True, help="Path to trained models")
    gen_parser.add_argument("--key", "-k", default="C:Maj", help="Key to generate music in")
    gen_parser.add_argument("--notes", "-n", default=256, type=int, help="Minimum number of notes to generate.")
    gen_parser.add_argument("--output", "-o", required=True, help="Output of generated MIDI file.")

    args = parser.parse_args()
    if args.command == "train":
        train_directory = args.data
        data = TrainingDataProcessedInfo()
        data.load_training_data(train_directory)

        model_output_dir = args.output

        # Train all the different models
        for model in args.models:
            if model == "chordhmm":
                print('Training ChordHMM...')
                chord_hmm = ChordHMM()
                chord_hmm.train_model(data.notes, data.beat_chords)
                chord_hmm.save_model(f"{model_output_dir}/chord_hmm.pkl")
                print(f'ChordHMM finished training.')
            if model == "bassgen":
                print('Training BassGen...')
                bass_model = BassGen()
                bass_model.train_model()
                bass_model.save_model(f"{model_output_dir}/bass_model.pkl")
                print('BassGen finished training.')
            if model == "rhythmhmm":
                print('Training RhythmHMM...')
                rhyhm_hmm = RhythmHMM()
                rhyhm_hmm.load_data(train_directory)
                rhyhm_hmm.train_model()
                rhyhm_hmm.save_model(f"{model_output_dir}/rhythm_hmm.pkl")
                print('RhythmHMM finished training')
            if model == "ornamentmcs":
                print('Training OrnamentNoteMCs...')
                ornament_mcs = OrnamentNoteMCs()
                ornament_mcs.train_mcs(data.ornament_groupings)
                ornament_mcs.save_model(f"{model_output_dir}/ornamentmcs.pkl")
                print('OrnamentNoteMCs finished training.')

    elif args.command == "generate":
        model = args.model
        num_notes = args.notes
        key = KeyTiming(args.key)
        output = args.output

        # Load in models
        print('Loading in models...')
        chord_hmm = ChordHMM.load(f"{model}/chord_hmm.pkl")
        bass_gen = BassGen.load_model(f"{model}/bass_model.pkl")
        rhythm_hmm = RhythmHMM.load_model(f"{model}/rhythm_hmm.pkl")
        ornament_mcs = OrnamentNoteMCs.load(f"{model}/ornamentmcs.pkl")
        print('Models loaded.')

        print('Generating song...')
        song_generator = MusicGen(chord_hmm, ornament_mcs, rhythm_hmm, bass_gen)
        song_generator.generate_midi_score(key, num_notes, output)
        print(f'Song generated.')