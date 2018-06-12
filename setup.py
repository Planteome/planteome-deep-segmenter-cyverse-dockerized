
# install script for PlanteomeDeepSegment
import sys
from bq.setup.module_setup import python_setup
from bq.setup.module_setup import read_config
from bq.setup.module_setup import docker_setup


def setup(i_params, *args, **kw):
    print('args={}, kw={}'.format(args, kw))
    python_setup('PlanteomeDeepSegment.py', params=i_params)
    docker_setup('bisque_uplanteome', 'PlanteomeDeepSegment', 'planteomedeepsegment', params=i_params)


if __name__ == "__main__":
    
    params = read_config('runtime-bisque.cfg')
    if len(sys.argv) > 1:
        params = eval(sys.argv[1])

    sys.exit(setup(params))
