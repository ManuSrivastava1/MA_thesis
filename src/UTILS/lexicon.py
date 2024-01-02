import codecs,pickle

class Lexicon:
	def __init__(self):
		self.frameLexicon = {}
		self.FELexicon = {}
		self.frameToId = {}
		self.idToFrame = {}
		self.FEToId = {}
		self.idToFE = {}
		self.frameToFE = {}
		self.source = "NA"

	def get_number_of_frames(self):
		return len(self.frameToId)

	def get_number_of_FEs(self):
		return len(self.FEToId)

	def get_id(self, frame):
		if frame not in self.frameToId:
			print("Unknown frame", frame, "assigning id=-1")
		return self.frameToId.get(frame, -1)

	def get_FEid(self, fe):
		if fe not in self.FEToId:
			print("Unknown FE", fe, "assigning id=-1")
		return self.FEToId.get(fe, -1)

	def get_available_frame_ids(self, lemmapos):
		return [self.frameToId[x] for x in self.frameLexicon.get(lemmapos, [])]

	def get_all_frame_ids(self):
		return list(self.idToFrame.keys())

	def get_frame(self, id):
		return self.idToFrame.get(id, "UNKNOWN_FRAME")

	# Load from pre-defined lexicon in format [frame \t lemmapos]
	def load_from_list(self, src, allFN):
		# LOAD FRAMES
		with codecs.open(src, "r", "utf-8") as f:
			frames = []
			for line in f:
				frame, lemmapos = line.strip().rstrip().split("\t")
				self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
				if (frame != 'Test35'):
					frames += [frame]
		with codecs.open(allFN, "r", "utf-8") as f:
			for line in f:
				line = line.split()
				if (line[0] not in frames):
					frames += [line[0]]
		frames += ['Root']
		frames = list(set(frames))
		self.frameToId = {frames[i]:i for i in range(len(frames))}
		self.idToFrame = {y:x for (x,y) in self.frameToId.items()}
		# LOAD FEs
		FEs = []
		with codecs.open(allFN.replace("Frames","FEs"), "r", "utf-8") as f:
			for line in f:
				line = line.split()
				if (line[0].upper() not in FEs):
					FEs += [line[0].upper()]
		FEs += ['NONE']
		FEs = list(set(FEs))
		self.FEToId = {FEs[i]:i for i in range(len(FEs))}
		self.idToFE = {y:x for (x,y) in self.FEToId.items()}
		self.source = src.split("/")[-1]
		# LOAD FRAME TO FEs
		fr2FE = pickle.load(open(allFN.replace("_AllFrames",".Frame-FE.pkl"), "rb"))
		for k,v in fr2FE.items():
			self.frameToFE[self.frameToId[k]] = []
			for fe in v:
				self.frameToFE[self.frameToId[k]].append(self.FEToId[fe.upper()])
		self.frameToFE[self.frameToId['Root']] = []

	def is_unknown(self, lemmapos):
		return lemmapos not in self.frameLexicon

	def is_ambiguous(self, lemmapos):
		return len(self.frameLexicon.get(lemmapos, []))>1

	# Load from training data
	def load_from_graphs(self, g_train):
		frames = []
		for g in g_train:
			predicate = g.get_predicate_head()
			lemmapos = predicate["lemmapos"]
			frame = predicate["frame"]
			self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
			frames += [frame]
		frames = list(set(frames))
		self.frameToId = {frames[i]: i for i in range(len(frames))}
		self.idToFrame = {y: x for (x, y) in self.frameToId.items()}
		self.source = "training_data"