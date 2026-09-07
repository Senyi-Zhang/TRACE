"""A flask server that uses the SPRING model to parse natural lanuage into AMR
graphs.  This service also optionally caches the results.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable
from dataclasses import dataclass, field
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, Any
import logging
from pathlib import Path
import textwrap as tw
from flask import Flask, request
import json
import torch
from penman.graph import Graph
from zensols.persist import persisted, Stash
from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(
        description="A service to predict AMRs from text sentences.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help="Endpoint bound port.")
    parser.add_argument(
        '--datadir', type=str, default=None,
        help="The directory to store cache files is provided.")
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument(
        '--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument(
        '--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument(
        '--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument(
        '--max-tokens', type=int, default=200,
        help="Max number of tokens allowed before creating error.")
    parser.add_argument(
        '--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument(
        '--only-ok', action='store_true')
    return parser.parse_args()


class Batcher(object):
    """An iterable that creates sentence batches.

    """
    def __init__(self, stash: Stash, sents: Tuple[str],
                 batch_size: int = 500, max_length: int = 200):
        data: Tuple[int, str, int] = []
        sents: Iterable[str] = filter(
            lambda s: len(s) > 0, map(str.strip, sents))
        errors: Dict[str, str] = {}
        cached: Dict[str, str] = {}
        for idx, sent in enumerate(sents):
            if sent in stash:
                cached[idx] = {
                    'graph': stash[sent],
                    'status': 'cached'}
            else:
                n: int = len(sent.split())
                if n > max_length:
                    snt: str = tw.shorten(sent, 80)
                    errors[idx] = {
                        'status': 'error',
                        'error': f'Sentence too long ({n}) tokens): {snt}'}
                else:
                    data.append((idx, sent, n))
        self._data = data
        self._batch_size = batch_size
        self.errors = errors
        self.cached = cached

    def __iter__(self) -> Iterable[Tuple[int, str, int]]:
        batch_size = self._batch_size
        data = sorted(self._data, key=lambda x: x[2], reverse=True)
        maxn = 0
        batch = []
        for sample in data:
            idx, sent, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)
                if 0 < curr_batch_size <= batch_size and \
                   cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)
        if len(batch) > 0:
            yield batch


@dataclass
class AmrParseService(object):
    """The service that uses the model to parse sentences into AMR graphs."""

    args: Any = field(default=None)
    """The parsed command line arguments.  See :func:`parse_args`."""

    @property
    @persisted('_stash')
    def stash(self) -> Stash:
        """A stash that caches sentences (keys) and their respective AMR
        prediction results (values).

        """
        if self.args.datadir is None:
            from zensols.db import DictionaryStash
            return DictionaryStash()
        else:
            from zensols.db.sqlite import SqliteDbStash
            from zensols.db.stash import DbStashEncoderDecoder
            db_file: Path = Path(self.args.datadir) / 'graphs.sqlite3'
            logger.info(f'caching graphs at {db_file}')
            return SqliteDbStash(
                encoder_decoder=DbStashEncoderDecoder(),
                path=db_file)

    @persisted('_model', cache_global=True)
    def _get_model_tokenizer(self):
        """Create the model and tokenizer.

        :return: a tuple of the SPRING mode, tokenizer and CUDA device

        """
        args = self.args
        model, tokenizer = instantiate_model_and_tokenizer(
            args.model,
            dropout=0.,
            attention_dropout=0,
            penman_linearization=args.penman_linearization,
            use_pointer_tokens=args.use_pointer_tokens,
        )
        device = torch.device(args.device)
        logger.info(f'loading checkpoint {args.checkpoint} to device: {device}')
        model.load_state_dict(torch.load(
            args.checkpoint, map_location='cpu')['model'])
        model.to(device)
        model.eval()
        return model, tokenizer, device

    def parse(self, sents: Tuple[str]) -> Dict[int, Dict[str, Any]]:
        """Parse sentences into AMR graphs.

        :param sents: the natural language sentences used for model prediction
                      into graphs

        :return: a dict of sentence index (for ordering) as keys and values with
                 graphs or errors with model status as values.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing: {tw.shorten(str(sents), 60)}')
        args = self.args
        model, tokenizer, device = self._get_model_tokenizer()
        iterator = Batcher(
            self.stash, sents,
            batch_size=args.batch_size,
            max_length=args.max_tokens)
        graphs: Dict[str, Dict[Any]] = {}
        graphs.update(iterator.cached)
        graphs.update(iterator.errors)
        for batch in iterator:
            ids, sentences, _ = zip(*batch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'parsing sentences: {sentences}')
            x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
            with torch.no_grad():
                model.amr_mode = True
                try:
                    out = model.generate(
                        **x, max_length=512,
                        decoder_start_token_id=0,
                        num_beams=args.beam_size)
                except Exception as e:
                    for idx, sent in zip(ids, sentences):
                        graphs[idx] = {
                            'status': 'error',
                            'error': str(e)}
                    continue
            for idx, sent, tokk in zip(ids, sentences, out):
                graph: Graph
                graph, status, (lin, backr) = tokenizer.decode_amr(
                    tokk.tolist(), restore_name_ops=args.restore_name_ops)
                status_str: str = str(status)
                is_err: bool = 'OK' not in status_str
                gr_str: str = None
                if not is_err:
                    graph.metadata['snt'] = sent
                    gr_str = encode(graph)
                    self.stash.dump(sent, gr_str)
                graphs[idx] = {
                    'model_status': status_str,
                    'status': 'success',
                    'graph': gr_str}
        return graphs


# Flask server instance
app = Flask(__name__)

# service instance
service = AmrParseService()


@app.route('/parse', methods=['POST'])
def parse_amr() -> Dict[int, Dict[str, Any]]:
    """Service endpoint that takes a JSON request (natural language sentences)
    and returns a JSON response (graphs or errors).

    """
    req: Dict[str, Any] = request.json
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'request: {req}')
    if 'sents' not in req:
        return {'error': "Expecting key 'sents' in request"}
    amrs: Dict[str, str] = service.parse(req['sents'])
    return json.dumps({'amrs': amrs})


if (__name__ == '__main__'):
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    service.args = parse_args()
    logger.info(f'starting service on {service.args.port}')
    app.run(host='0.0.0.0', port=service.args.port)
