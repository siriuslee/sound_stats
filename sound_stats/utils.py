import os
import uuid


def get_temporary_directory(root="/auto/tdrive/tlee/tmp", type="raid"):

    if type == "raid":
        return os.path.join(root, str(uuid.uuid4()))
    elif type == "ceph":
        raise NotImplementedError()

def dump_wavefiles(waveforms, directory, sample_rate=44100):
    from scipy.io import wavfile

    if isinstance(waveforms[0], tuple):
        waveforms0, waveforms1 = zip(*waveforms)
        waveforms0 = dump_wavefiles(list(waveforms0), directory,
                                    sample_rate=sample_rate)
        waveforms1 = dump_wavefiles(list(waveforms1), directory,
                                    sample_rate=sample_rate)
        waveforms = zip(waveforms0, waveforms1)
    else:
        for ii in range(len(waveforms)):
            if not isinstance(waveforms[ii], str):
                filename = os.path.join(directory, str(uuid.uuid4()) + ".wav")
                wavfile.write(filename, sample_rate, waveforms[ii])
                waveforms[ii] = filename

    return waveforms


class ClusterRunner(object):

    def __init__(self):

        pass

    def run(self, cmdList):

        pass

class SlurmRunner(ClusterRunner):

    def run(self, cmdList, **kwargs):
        from slurm_tools import slurm_sbatch

        kwargs.setdefault("mem", "8000")
        kwargs.setdefault("partition", "regular")
        return slurm_sbatch(cmdList, **kwargs)


class SavioRunner(ClusterRunner):

    def run(self, cmdList):

        pass
