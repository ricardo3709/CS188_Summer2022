#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('QlpoOTFBWSZTWaCejNwAPDRfgHkQfv///3////7////7YB18Ek+62cJjqdcoB2pjYAZmw24BtgJEEBBBK0A0OI6oHAEAABKqDTQdUVjQG2t3Y0aYSggkaamyaaaI0nqnp6ajGoynqYnqA/Smg0DI000P1TagBpoTICQJqaYnqp5T1GATPSj0jAmJkyZNAMENNA4GjRiDRpkwgxAYjE0aNGgDTTQAAABJooopkFBD9UaepiAepiaA9Q0AAxAAMTQaDgaNGINGmTCDEBiMTRo0aANNNAAAAEiRDQBAE0EwjTSYEno0J6epNpTExDxTQDTRtPqeZD4xPqgfQFn3WF/oan4Ws+2lQY/92VRFUYjIxUB7bGKf2Up5tnxaxD+9r6JCsD5k93pQ5aWI+UbD8Dz86z/p8IiTxb0Q2lj+Lmnh4xHiFYsWIIsARWCQYs6gn59cw+/j3vs9xtD6CB80dO+CEVQHXn22ZsPszKu9aV7qeG4W5Y8HFY0V7at6+3uss/a6z6MuHtzPk92JkKN+tyRCanfaE40MPegl3o5nEE2ohwSAUFYgrIosRAWKoqgxFgCMFRiwURWeP2vpfNPmn8/v9wz1/UPwMN8PPx3QtobUoUGCCVrcHp0nqhjX7Kcbw7ahrNrdityXkitLJWDtsu4HH5L4yX00Q6g72XBjyuSZsBVq4mem2ukwZj/rOCS8MLqsHK/GzOtMaY2JsIlM9mcqZhRdVgK7YvMiEwuxcA33PJuMuzcI3MuF5z6LjOVBBTbcZWX7O2+b0jHVZbzlWrTLIYxjX1QQ28HGdehFXnlL81o3Wewd1rbeiu9+ap8g/ZVCF0hpw0YGJyYoGQekOU9UcrjV/jrikij6AraHEx4EFCCQQCUBKAlG8INZxfOL07VVlLu++iLXIqq5CqCo3tgbmq280s5wppTSkqI7lDG4OsuMG15tC8LOVrSNLSXXRJZVuGZC0qKLy48PXvGKqFkvJqqqaM4w4kDiKqr9Pgc70Isixe+x6eAcURIiqgbmXSpWtF2FTWDThHy9uBNTsbFpW40msW33eZq25acL8HWSl+w13lWNvOnZr2gLDjGM/EOPzD0hqkzCt0jjIvnzs5zCnzX6PB3LMXHLplWJga7IQLD+1S1fiP5wDQg6ToyZm6t1UmCdxY6WrEdEV+P17I+jTy+DDgMUlypXfonFMgU5qoAYVRPFdXi4H4c2m7JHuwFS9arq63XSCabGmO3keES8eCq8dMj/Hi9vS8eg73zpx829mTW+hFNMprr43OKufXkvOpsjtKLPKUs5CepRs7KpggbnNxiefvg1TgQ5kbJEK6IbEoqhNzcNpR2YX/j7fsl5f//o2IKasM8dIbEx5snKEm/7QyCbc9aCRJneFlEKN743EDiPbaXZ/JMeDUDxUdWfMsGR1ytmtUlCPl9DTgb+tgolCU28LXW8o8WKenNKrhWTlAyw9yggMxMv41MeZgLECQbEu9m3L3pPjU03I5GdFQpkPlSqlzRQWYdNZYUrm4pQrArmgaGVkSlujfIC8YaW/Rqqtc87oL571sBGFVfR2F7B/E1tUBBAgRdbWFdOmNVZLBUylJyBeiOpgiyhqwwEjSHiur1kREdz9N9r3pi+SMwsUhtFUokCo50qdkcVF6TVXBuqwI7LbmIftS1NcyFvrI39Q2tv83n/jHNNK/iX5vBbftNrnof37PQX7YaufnLJY2LbodX20oGeuzWPwZYBlz5FsRrkIl7Xd5Koq7Q4ZOxWJoQdzVhRNFurzklfqLAZICokGDUvcsUYqr+d2KfoIwO8pUsBYiP4drG5c+N47nTUdcC+ztuTr729ee608++ng7tX2Ktzbw7jip5DfGGHCgr6sZHgrWNFtdRyUWSKWk9+GqwStGwNAyCx3QCg0EUfr22ORzNXDQ3qwhd3NlLeysec8Tsrq7IjtMI5XRPbAEdDvOmL2IuJ8JKJTNpm3DF9t8s2WclBYbNwa3HepnmYvBPWTkYSh6nqr62m1Jg2D+f/2z5bdR008jEtTCGPIh7c5KTWLqbpEc/ZXOsdWqIZRqHE+6G5OwT87CsUZjkfVapS/Rq7C93HoUhhQdSLOUhvV8LRlaVw9CrhRWgWFez+QVd1pTcKKWdhwrTLNEhRrA4+tlrzkFb3DtQBwMpsGJIcW/WwfSLeJCqqctIV+Ch5yOgeMaLvGh4nROU3dOvQVBf2672PPHyqund67HPv7B/SBrKuGU9TQVitd3ZyDva7ogIGdZWrqIMDR6m3lXYQp/u8J1IKkBp+u4CLtjswB2O2LMBFiDurxwXm7LaIqsN4EWe1GMYxTVnFreHlznJ0VWEJsaJXYc1tl2H7KRoNYF2jCq1NNZa7IqA+I5MxqIKzZjfDhhcgMFfq08z3gh22ETvllKFCDPZu3VLDdXZeZYFcn7/X4c0VffmP7QWmfiWINgKsQ115xk/mQM9ncfJB2rx8fPL4j9rSzdZ4VVTzKgkL4A4GTXpsEQn6bSOj4wEPBznb4r7gOJdTCK8RtyXR8CJtMbfuGXtYUilfjr+v/OyKfXziPz3bQe0WJbksjxvuaCVdWGnNziMO2kTw6PZmtc+bF2kOm7Tow6MBRa9lJBBHEcc2ifJyu4wLgXwo3Xi40LhxNl3KnbzrTb1QdrWsoN88ZWbDi44kIKydx2IQXmnAM0oVLjhF1j0da+Zc6isx5NrCXZeofnmzNkeCoQIOvyzwyCDjeVWBctPmcQTwqIkWFTalB2dHX+vZkw4RDxOkQg0aW9gssrDLss91PqdW/YXSraVPdbSV+p787sJ1zLpbFz26twFMuQRX2xqdX+PPTkvO6z7x1yct4epwQKUlqqchWFZWmqqzqaBQw8zhgXWaAz/U3ws1PVyfUkIrKI2UCWGDOF5eekXqRFVqF3hnVDJk+AiHzW0OG6worJUr0qOsFpEPGby7McUmWWyt2YHepss25NN0JzpHz/StHZq8mvX+F5/gACIVaatG6H182zI+nP/oABIzl/Tl+YACR76H1/sj5Ug696t6XulKTVNwVxZw5dk4UvLx460VKSqJS85rreF5y1ovFo2nr04ZRKrWr1yJLsIhimJtQVbIgJFdEUtFGEsjGL3kjH+rs6HELEPNMHatjQYUksAQYAsfEB0sizd4FKw0LroatW3YmgKOJRFULJtDUjUVROuJJ1reCtmXGNdgrWolFWywxpIoAiNJRjP04h+W+Hu9nLp+4ABEPv5zzUgtAACIfA/h4aj2gAJE+HV84ACRJcN2rYHAibgkwicQ5E5uc25WRDXJkE2ltRNbxxwebXV1Mi3UGmMJsxmYus5ubIt3KVbBSuOfC8UeFS2nBxWmAozQhVHCTSSIy2I95xUtA4QpUfaf+fH496M81UGewNhl1GxY6bUaSs9lrUMUYYmElVkXVpQnrS9GsjDmZUtKFLnYqZXKlAYYIoFEmmShNkCnifP7gAEjt8AAEjp+YABIssOz578UdbgIMxQMcMgFviCc4IiTJNJjj4E+eRIIJJMBZCgwYbDjSAIw70OUpW2aEOCRGSYpZEQlMBjEgjAgbh0eHIAjOFKSIpAQQQJjSP+bj8MKWFsNkQUKAkMaSiQ5EgGMBSIIBjA0qNhb+sABI9WHp6/1AAJGWNs+enT58R4+F3xAASJxP8QAEiIeqOXOZevwAASLKudlP7AAJE7fdAjf6PZVZ/UABI2WUYcH7tQACREuIACR2LrDnY223a0opsmbarbtQqDMGrcmudv1/PDk6UiMKiCJ+aHyjCIwiyTnFFwYUiMOHB00BiTmPJYqwFWNoxHkOEMUNIIYgBTHeyrY1i21BQgjFIDEEFozpOkEQOAZcYkgiEiJBBBjA7w6otV5FJphBpsorQ++ABIfH+EABI1l3R5kfk23cxR8WfrnnB6HsqfoqWrZOoY3P5jqzHWLNtNpH2bj8ccc4m63+ER/KFrGHG6AG7LZSsdDFePxD159C3FLKg+KHWuHsR9lq+988fEBjdduJP739Pwj1GF4yTp9xBbMtwJfDOpGRfKVKQa9v8xxA+ZY1biSGuXLfz9sxp+SIbdCAYnIkMgnQgknm1ZTWWCosjuUM/6cskFM1fklkCsH7X5eN9L08hj0XdnE5MalhijbW2nCyMFjJTTEHW2FiThOZaEsdFbVrRG6mqRgkkY4SGKYiObLLFSchxaHBkEQQxrKUEGnMRAwCIcIl+639mGZy3I+1HBfiHkhHm3fgdiZ6kHPhZNoKhsYoIwV+ItiWw2ENNg001r+RbaCQTLKVKNGIyjAxUByS2I507EZp0Koa5l1KUWI+t11TJ6awKskVTiWPu6gctIZ3AHnRC4cgIAzEHanHR31EgVixnMU4VcVB6EXosPClxzqOsi/K71oIVq6AnDsWKBs5mXdgPRh2cJgF5jY2WZpjddJGOH2NcTfEZltx4YmwMznMC6uj+b2/nYUQD32q0ykZ8GZ1BE0EK5XK/ujmvgSm1usPquPAzAASJmYXxJWcZEkTLGSO7aSqVlWB0samChogMbFy7xRWVq0MJix+Mq0T2oDDY680rrlRZsNmo3/G57M5EUZD35kmA+moI1q/+ctLGjddOarULYEtauEafogca3OC9lzcyO8ABIkuMpovoYS1x0yRW+h9PqxLeWe9s+odV/+J9rxA498TSTBYgvow3Yh1dndxXJcOC0/j9XSidRvQF6R5BNkxx5RCm0mhpSyA5jYk60g7dlc53ZDDSruxvsYNhyrQskVWm4sDmqM3RdtkwDoZ9GoMRGeNjs56vkAAkbDqzG90pSZ1+Es5Kb8LY0tJ+ofXbAX3cLaVAMy2SW50hbRD3XTQ+vIzsnoaIwGKpUIhAS0q0Kjj1JiGo7rqE6AQrCBIaxkWaN98zYChyypYFEDjtKFDShIexrnLRmrCKmdnrr7wAAEQSTZIoXLWDWchVASF7cp7LJSDagj+RKlEdeqb299Uq3R4yUBVV1aAYIOgR3eqEDQY24CSOe4NfbxOSJ81aRbzrMsu9hNsIA68+rs59XMs0QH8dSzPo4VW0swu4pbHRNCk2VlpRGtAzbo0XXi0ycV0aTUokEiZEEYo0uUwsFG/G6gtwDVRdtfkwgTjIuriCoM0R7Ii3EZnAoefSIS23Jitpxf6OQ1iMTmb8DYapBVrmRZs2Osk0qnhwaj7TpxgrC0VEBdS2LEmSumIukAXliD5Vd1gAUooLuYWfIYyryJiHUR0ro20khbU2JNoaY0MSGT+qqD28u5U6JRL5RW0P9uHaTTw4lYecWITCfT2FDDnWuS81RCgwNJKW/2oJTn6ypE06WEEvQmDbGhPFdeY6kGR0gAJFi6cIJqxU17JUmTaC/vUDMfShgPCgyAp8kqW2B/AuML+/OVbSuQa/Hnr08ePO8xwJ5Q270DvC4R73vPPeTvjm1xdzZTNOW95o97TGYZxjFogtggZGyQMllRBbJN0WJS3Ra1tbEtc4bENRTZVWgxKDDgZwhG0QKoMlKhQbaIyJgNCqwUKRMrFiIFhzYtq4aa1tiWlXFo0mDI0KZY0H08eEAQfdZ/SR9d86T6aeScEMSO4XGLNxZICViU2jiM9m643+dgM3EQ02n+gACQ5XZ1BA/axEBURBGRs6fUTRWrGRamCIIfh0XbjYCqrNNqJM9MLWNP+XlDQABIdytdVflXAEer0emrycUjQRfnwC/fmG34NJQ35iEcrVpBUZW+2rYnYBelX2Js7yn2NbWA2kTTEHsaVYINADMSXAFxOKpdSb0R/SL0d3/Z5g3GiH1npW7DZvG0TZwlxciCTxHFYxpkpQ9l27b50GeXfasS1FIzTQ2m2IGNjTQ/UIqP9IMrEaJBzvyHDdg0dWHfqO1UtLxbkfC+RIGA8YDUU2dqXNPK8tnQ3anLXhSxqRAACR1mILLJyx7YnOFzzqBmXOKSRUHWWO23dPfFyy5gAEi25zRf52sxqGoEDUPqggZBEn3ov89gSDaUG9TQi5hcj+FIvJqvVCmxK3QwmkTfGg6Yqa5zlby8+NZw4Zi2/oAAkM0YG2CBbpRvIhbLuVrR24abSDLNFR3PGydXAwKGsyZms5MD16oA1ogOFQdRZdIzrmSzcKZnUiJT5AAJGoNfLI8jNW26D09ckjdUs9Iym9oSiC6FDaQNJgMjixpsNd0bN9dOBeX7sCXQvnisMkcgMgGK3O5BwoP30CbwAPnGSsmDZn9GO18S9ctnptvGPiCgntaPvLeDitLsgxijGIoiIge9eY5U/kgVifHz5DdoH0fgl5CHX8VcEwZQtYEpCYVl2XRSnLCbnEPebLkVANo9IeucUYHtKQ0ApLIz5OG+UTiEh1AdFkmWYil4Cs4ASr373uGGoYGi7dKXJpGaYUBQD1ScYIVl+cknD7AAEgwhOs719Tsety98bZwAAkcEt3Z+0keuYQtQqhtuXUbTo3G+n4t2QIwBQecEeIKDtsti6zWz9KrPVpG9boUtT0aROfE0aBsSJShImHCz3HFfvssViJMzC6PX5dnDjVd7KmgUFpbsmqwoh0rXp7xVhm0i4LQRWi/fiHVXXRI5mT9a1mrQpP8hzmBvXPzdNFU8QZhPjwCp4+U6NIGpQ5W0NtjHWyY+f3k9C449E7Rof3eTlhqbs7AAEjFHZ2bStixzkJE6WvZSbXJsYYuy2zckWowAqX178LbDFAHre2eNZ+QAEhjdIe0z7vQ9mSKPwjbask+CUfVsHbYTJhCTEaWWy2AAJFICuIUK4D3AH4MwxQF/SATttWDyGSTolwyrlVBVFsQXIw7RZCq5sD2FETOkPmuWQx1Ywkl1O7jlXoDV/dVKcypMrUHoNVaKkh3pgDADVQVBZsLu7rVeDrSv30wvRzqAxaaegTAgLZgoYZNNO/0fbQOKEmAghAvlwq9C/uiEWQMED2rUvSGUclRUkwdzNdLLhk6vNJ5RY6lUzbKXdATmzDM0ogtgqgNdFvNtG4KVM2ZGQAolggU/lYbIglocNDO6m2yd+FyvrCQq80np5HzOeYbyKScDGQEKs8jXoxt8o7bq7rETDC6fWgJIM1uCsnUfC27unhckUvhMaAbV+7PbmsNhsj3jnQ1CysBsTQiSKgAKE6CrpV5oAoA4C1uxouOC3xYX4xPNCHpbhg+3ggmSfi4uuV0TvLnyRLRqmMVA+m6Exg2p+w+on6T1mL3xeMjFqZnh49rx53g+PB1zZeIuh1UnOcQSZft2WLXKwyVFizhXSmb5ponE4bFNqjaqsWZ1URDMNoJCrB/waF1Qgn5yp4vNleyQG9cPFO+29ac546VGLHXMrK8qb28mO7GG6XlD2IbrCpxrW2iJ47a869dp411rwzPY8ZytDiPuYKRtgqIPFGtLxUMgqg+i2CkebBacnL6XoHLfo7dVrwa1znuvI5xPW9cdy2tpQYIJbSqIraVVVVZUDzyu8VHA4Ut5fYJCOvPb1xzjRZW10chtJ0c3J3hYWysdbrhwwfUZnDjx6d75WOLPLDBWWeuOwqQOhsZIlAboDch1szQQmMGggxOnHAdbwmgtgO8eOcnRh1Y86XshGCO8KE8rVyAKyB7xpv2095sCrANRn05KkUNeMJeW010vR4sIeXcwfmwxCq28EUSd7GpIDhVNhOVIcCNCyJ3p7bthluC5UUVAmgZIFmW0zw49Pn41+F01vLvhJQAJDoS+iaGjD0BQOvmPNxRJK90Q9uJkNYsRAh7IkPkC0sTkAE8DC9bi1U4iGIuqDlIAnjouOZKFM+4RGcH5jrAASP1BFk8Vz384MXseDsOVaGv4UM/QjQd3QooSyPHNwA4wjMCXELLU1wte3LRnFKbSvei/mMo+5g2iFQuWm+mnT06rAAEhznlf1O+OhN7adjqwKnLKd/eTlgwmvP+8ABIxrrN5YLmmUtUJTFhYEtDkffr/mAAkY0F72kw6Os7I3cZTR1nnve8hPLIYLO7kMQYVn5SfI6ny9ssFUaI6tAAEjeiov+myGavExGoRtL2gfRcAAkZGZhYVethwCj/Fx7X2P8iGvw9/lpdV8Ow/Nevev6L/F3dkmkZwHHJj1oH/i7kinChIUE9GbgA=')))

