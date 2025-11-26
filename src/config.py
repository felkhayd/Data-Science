import re

ACTION_RE = re.compile(
    r"^(?P<base>[^(<$1]+?)"              # action de base, ex: "Création d'un écran"
    r"(?:\((?P<ctrl>[^)]*)\))?"          # (controller/écran)
    r"(?:<(?P<conf>[^>]+)>)?"            # <configuration>
    r"(?:\$(?P<chain>[^$]+)\$)?"         # $chaine$
    r"(?P<edit>1)?$"                     # flag édition "1"
)