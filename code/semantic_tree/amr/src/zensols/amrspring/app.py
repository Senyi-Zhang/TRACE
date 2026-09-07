"""A client and server that generates AMR graphs from natural language
sentences.

Citation:

  `Bevilacqua et al. (2021)`_ One SPRING to Rule Them Both: Symmetric AMR
  Semantic Parsing and Generation without a Complex Pipeline. In Proceedings of
  the AAAI Conference on Artificial Intelligence, volume 35, pages 12564–12573,
  Virtual, May.

  .. _Bevilacqua et al. (2021): https://ojs.aaai.org/index.php/AAAI/article/view/17489

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Dict, List, Any
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
from pathlib import Path
import requests
from requests.models import Response
from zensols.util.time import time
from zensols.config import Dictable
from zensols.util import APIError

logger = logging.getLogger(__name__)


class AmrServiceError(APIError):
    """Client API errors."""
    pass


class AmrServiceRequestError(AmrServiceError):
    """Errors raised for the protocol transport layer."""
    def __init__(self, request: Dict[str, Any], res: Response):
        super().__init__(f'Error ({res.status_code}): {res.reason}')
        self.response = res


@dataclass
class AmrPrediction(Dictable):
    """An AMR prediction or error.

    """
    sent: str = field()
    """The sentence that was used for the prediction."""

    graph: str = field()
    """The Penman formatted graph string."""

    error: str = field()
    """The message of the error or ``None`` if the prediction was successful."""

    @property
    def is_error(self) -> bool:
        """Whether this prediction was an error."""
        return self.error is not None

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'sent: {self.sent}', depth, writer, max_len=True)
        if self.is_error:
            self._write_line('error:', depth, writer)
            self._write_line(self.error, depth + 1, writer)
        else:
            self._write_line('graph:', depth, writer)
            self._write_block(self.graph, depth + 1, writer)


@dataclass
class AmrParseClient(object):
    """The client endpoint used to communicate to the server that parses the
    AMR graphs.

    Citation:

      `Bevilacqua et al. (2021)`_ One SPRING to Rule Them Both: Symmetric AMR
      Semantic Parsing and Generation without a Complex Pipeline. In Proceedings
      of the AAAI Conference on Artificial Intelligence, volume 35, pages
      12564–12573, Virtual, May.

      .. _Bevilacqua et al. (2021): https://ojs.aaai.org/index.php/AAAI/article/view/17489

    """
    host: str = field()
    """The prediction server host name."""

    port: int = field()
    """The prediction server host port."""

    def _invoke(self, data: Dict[str, Any]):
        endpoint: str = f'http://{self.host}:{self.port}/parse'
        res: Response = requests.post(endpoint, json=data)
        if res.status_code != 200:
            raise AmrServiceRequestError(data, res)
        serv_res: Dict[str, Any] = res.json()
        if 'error' in serv_res:
            raise AmrServiceError(serv_res['error'])
        return serv_res

    def _parse(self, sents: Tuple[str]):
        with time(f'parsed {len(sents)} sentences', logging.INFO, logger):
            res: Dict[str, Any] = self._invoke({'sents': sents})
            if 'amrs' not in res:
                raise AmrServiceError(f'Unknown data: {res}')
            return res['amrs']

    def parse(self, sents: Tuple[str]) -> Iterable[AmrPrediction]:
        """Parse ``sents`` and generate AMRs.

        :param sents: the sentence strings to use as input

        :return: an iterable of predicted AMR graph results

        """
        server_res: Dict[int, Dict[str, Any]] = self._parse(sents)
        if len(server_res) != len(sents):
            raise AmrServiceError(
                'Inequal sentence requst to response count:' +
                f'{len(server_res)} != {len(sents)}')
        preds: List[Dict] = sorted(server_res.items(), key=lambda t: t[0])
        sent: str
        sent_res: Dict[str, Any]
        for sent, sent_res in zip(sents, map(lambda t: t[1], preds)):
            status: str = sent_res['status']
            if status == 'error':
                yield AmrPrediction(
                    sent=sent,
                    graph=None,
                    error=sent_res['error'])
            else:
                yield AmrPrediction(
                    sent=sent,
                    graph=sent_res['graph'],
                    error=None)


@dataclass
class Application(object):
    """A client and server that generates AMR graphs from natural language
    sentences.

    """
    client: AmrParseClient = field()
    """The client endpoint object."""

    def _get_text(self, text_or_file: str) -> Tuple[str]:
        path = Path(text_or_file)
        if path.is_file():
            with open(path) as f:
                return tuple(map(str.strip, f.readlines()))
        return (text_or_file,)

    def parse(self, text_or_file: str):
        """Parse ``text`` and write generated AMRs.

        :param text_or_file: if the file exists, use the contents of the file,
                             otherwise, the sentence(s) to parse

        """
        sents: Tuple[str] = self._get_text(text_or_file)
        pred: AmrPrediction
        for ix, pred in enumerate(self.client.parse(sents)):
            if ix > 0:
                print('_' * Dictable.WRITABLE_MAX_COL)
            pred.write()
