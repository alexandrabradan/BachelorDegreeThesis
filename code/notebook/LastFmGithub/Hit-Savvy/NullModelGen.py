import random

__author__ = 'GiulioRossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class NullModel(object):

    def __init__(self, filename, out_filename):
        self.filename = filename
        self.out_filename = out_filename

    def execute(self):

        f = open("%s" % self.filename)

        adopter_to_slot = {}  # count all the adoptions of a good made in slot_x
        adopters = {}  # count total adoptions made by some adopter

        for row in f:

            row = row.rstrip().split(",")

            k = "%s,%s" % (row[0], row[2])  # good, slot

            if k not in adopter_to_slot:
                adopter_to_slot[k] = int(row[3])  # quantity
            else:
                adopter_to_slot[k] += int(row[3])

            ad = int(row[1])  # adopter 
            if ad not in adopters:
                adopters[ad] = int(row[3])  # quantity
            else:
                adopters[ad] += int(row[3])

        al_ad_slot = []  # transform dict into list
        al_us = []  # transform dict into list

        for u in adopter_to_slot:  # u=(good, slot)
            al_ad_slot.append([u, adopter_to_slot[u]])  # adopter_to_slot[u]=tot_quantity

        for u in adopters:  # u=adopter
            al_us.append([u, adopters[u]])  # adopters[u]=tot_adopter_quantity

        finale = {}  # count total adoptions made by some adopter of some good in slot_x

        while len(al_ad_slot) != 0:
            now_us = random.randint(0, len(al_us)-1)
            cur_us = al_us[now_us][0]  # random adopter
            al_us[now_us][1] -= 1  # decrement by 1 random adopter's tot. playcount
            if al_us[now_us][1] == 0:
                b = al_us[:now_us]  # get previous adopters
                b.extend(al_us[now_us+1:])  # get next adopters
                al_us = b  # remove current adopter

            now_ad_slot = random.randint(0, len(al_ad_slot)-1)  # random (good, slot)
            cur_ar_mo = al_ad_slot[now_ad_slot][0]  
            al_ad_slot[now_ad_slot][1] -= 1  # decrement by 1 random (good, slot)'s tot. playcount
            if al_ad_slot[now_ad_slot][1] == 0:
                a = al_ad_slot[:now_ad_slot]
                a.extend(al_ad_slot[now_ad_slot+1:])
                al_ad_slot = a

            k = "%s,%s,%s" % (cur_ar_mo.split(",")[0], cur_us, cur_ar_mo.split(",")[1])  # random_good, random_adopter, slot
            if k not in finale:
                finale[k] = 1
            else:
                finale[k] += 1

        out = open("%s" % self.out_filename, "w")

        c = 0

        for k, v in finale.items():  # k=(good,adopter,slot)  v=(adopter_adoptions_in_slot)
            c += 1
            out.write("%s,%s\n" % (k, v))
            if c % 1000 == 0:
                out.flush()

        out.flush()
        out.close()


if __name__ == "__main__":

    args_log = "adoption_log.csv"
    args_out = "null_model_res"

    args = parser.parse_args()
    g = NullModel(args_log, args_out)
    g.execute()
