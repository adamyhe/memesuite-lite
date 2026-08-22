# io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numba
import numpy
import pyfaidx


@numba.njit(cache=True)
def _fast_convert(X, mapping):
	for i in range(X.shape[0]):
		X[i] = mapping[X[i]]


def _fasta_to_flat_array(filename, alphabet=['A', 'C', 'G', 'T']):
	"""An internal function for loading a FASTA file into a flat int8 array.

	This method reads in a FASTA-formatted file and returns the sequences
	concatenated into a single flat array of alphabet indexes (-1 for any
	character not in the alphabet, e.g. 'N'), along with a cumulative offset
	array demarcating where each sequence begins and ends. This representation
	supports sequences of different lengths.


	Parameters
	----------
	filename: str
		The filename of the FASTA-formatted file to read in.

	alphabet: list, optional
		A list of characters to use for the alphabet, defining the order that
		characters should appear. Default is ['A', 'C', 'G', 'T'].


	Returns
	-------
	names: numpy.ndarray
		The names of the sequences in the FASTA file, in order.

	X: numpy.ndarray, shape=(-1,)
		A flat int8 array of alphabet indexes for all sequences concatenated
		together.

	X_lengths: numpy.ndarray, shape=(len(names)+1,)
		The cumulative offsets demarcating each sequence's span within `X`.
	"""

	fasta = pyfaidx.Fasta(filename)
	names = numpy.array(list(fasta.keys()))
	X, lengths = [], [0]

	alphabet = ''.join(alphabet)
	alpha_idxs = numpy.frombuffer(bytearray(alphabet, 'utf8'), dtype=numpy.int8)
	one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 1
	for i, idx in enumerate(alpha_idxs):
		one_hot_mapping[idx] = i

	for name, chrom in fasta.items():
		chrom = chrom[:].seq.upper()
		lengths.append(lengths[-1] + len(chrom))

		X_idxs = numpy.frombuffer(bytearray(chrom, "utf8"), dtype=numpy.int8)
		_fast_convert(X_idxs, one_hot_mapping)
		X.append(X_idxs)

	X = numpy.concatenate(X)
	X_lengths = numpy.array(lengths, dtype=numpy.int64)
	return names, X, X_lengths


def read_meme(filename, n_motifs=None):
	"""Read a MEME file and return a dictionary of PWMs.

	This method takes in the filename of a MEME-formatted file to read in
	and returns a dictionary of the PWMs where the keys are the metadata
	line and the values are the PWMs.


	Parameters
	----------
	filename: str
		The filename of the MEME-formatted file to read in


	Returns
	-------
	motifs: dict
		A dictionary of the motifs in the MEME file.
	"""

	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.replace('MOTIF ', '').strip("\r\n")
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = numpy.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.strip("\r\n").split()))
				i += 1

			else:
				motifs[motif] = pwm.T
				motif, width, i = None, None, 0

				if n_motifs is not None and len(motifs) == n_motifs:
					break

	return motifs


def write_meme(filename, motifs):
	"""Write a MEME file.

	This method takes in a filename and either a list or dictionary of motifs and
	writes them to disk in a MEME-formatted file.


	Parameters
	----------
	filename: str
		The name of the MEME-formatted file to save.

	motifs: list or dict
		The set of motifs to save. If a list, the name of each motif will be its
		numerical ordering in the list. If a dictionary, the name will be the key
		in the dictionary.
	"""

	with open(filename, "w") as outfile:
		outfile.write("MEME version 4\n\n")
		outfile.write("ALPHABET= ACGT\n\n")
		outfile.write("strands: + -\n\n")
		outfile.write("Background letter frequencies\n")
		outfile.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

		if isinstance(motifs, dict):
			motif_pwms = list(motifs.values())
			motif_names = list(motifs.keys())
		else:
			motif_pwms = motifs
			motif_names = [str(i) for i in range(len(motifs))]

		for name, pwm in zip(motif_names, motif_pwms):
			outfile.write("MOTIF {}\n".format(name))
			outfile.write("letter-probability matrix: alength= {} w= {} nsites= 1 E= 0\n".format(*pwm.shape))

			for col in pwm.T:
				outfile.write("{} {} {} {}\n".format(*col))

			outfile.write("URL BLANK\n\n")
		