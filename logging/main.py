import logging

import a
import a.aa
import b
import c

logging.basicConfig(
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
    level=logging.DEBUG
)

a.run()
a.aa.run()
b.run()
c.run()

