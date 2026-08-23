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


def _load_motifs(motifs, param_name='motifs'):
	"""An internal function for loading a dict of PWMs from a file or dict.

	This method accepts either the filename of a MEME-formatted file (parsed
	via `read_meme`) or a dict mapping motif names to PWMs, and returns
	parallel lists of names and PWM arrays in a stable order. Each PWM in a
	dict may be a numpy array directly, or any object exposing a `.numpy()`
	method (e.g. a PyTorch tensor).


	Parameters
	----------
	motifs: str or dict
		A MEME file to load containing motifs, or a dictionary where the
		keys are names of motifs and the values are PWMs with shape
		(len(alphabet), pwm_length).

	param_name: str, optional
		The name to use for `motifs` in any raised error message, so that
		callers with a differently-named parameter (e.g. `primary_motif`)
		can produce an accurate message. Default is 'motifs'.


	Returns
	-------
	names: list of str
		The motif names, in the order they appear in `motifs`.

	pwms: list of numpy.ndarray
		The corresponding PWMs, each with shape (len(alphabet), pwm_length).
	"""

	if isinstance(motifs, str):
		motifs_ = read_meme(motifs)
	elif isinstance(motifs, dict):
		motifs_ = motifs
	else:
		raise ValueError(f"`{param_name}` must be a dict or a filename.")

	names = list(motifs_.keys())
	pwms = []
	for name in names:
		pwm = motifs_[name]
		if not isinstance(pwm, numpy.ndarray):
			try:
				pwm = pwm.numpy()
			except:
				raise ValueError(f"`{param_name}` must be a "
					f"dict[str, numpy.ndarray], not {type(pwm)}.")
		pwms.append(pwm)

	return names, pwms
