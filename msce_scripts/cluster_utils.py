
import numpy as np

# -----------------------------------------------------------------------------

def read_cluster_info(filename = 'clusters.out'):

    file = open(filename, "r")
    file_info = file.readlines()
    file.close
    # print('  Reading {}: {} lines ...'.format(filename, len(file_info)))

    empty_cluster = np.zeros([1, 3])
    point_cluster = np.zeros([1, 6])
    pair_clusters = np.zeros([1, 9])
    trip_clusters = np.zeros([1, 12])
    quad_clusters = np.zeros([1, 15])
    site5_clusters = np.zeros([1, 18])
    site6_clusters = np.zeros([1, 21])
    kline = 0
    while kline < len(file_info):
        temparg_clust = np.zeros([1, 3])

        content = file_info[kline].replace('\n', '').split()
        # print('multiplicity: ', content)
        temparg_clust[0, 0] = int(content[0])    # multiplicity
        kline += 1

        content = file_info[kline].split()
        # print('diameter: ', content)
        temparg_clust[0, 1] = float(content[0])    # diameter
        kline += 1

        content = file_info[kline].replace('\n', '').split()
        # print('number of sites: ', content)
        temparg_clust[0, 2] = int(content[0])    # number of sites
        kline += 1

        if temparg_clust[0, 2] == 0:
            empty_cluster = np.vstack((empty_cluster, temparg_clust))
        elif temparg_clust[0, 2] == 1:

            SitePos = np.zeros([1, 3])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            point_cluster = np.vstack((point_cluster, temparg_clust))

        elif temparg_clust[0, 2] == 2:

            SitePos = np.zeros([1, 6])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])
            content = file_info[kline + 1].replace('\n', '').split()
            SitePos[0, 3] = float(content[0])
            SitePos[0, 4] = float(content[1])
            SitePos[0, 5] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            pair_clusters = np.vstack((pair_clusters, temparg_clust))

        elif temparg_clust[0, 2] == 3:

            SitePos = np.zeros([1, 9])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])
            content = file_info[kline + 1].replace('\n', '').split()
            SitePos[0, 3] = float(content[0])
            SitePos[0, 4] = float(content[1])
            SitePos[0, 5] = float(content[2])
            content = file_info[kline + 2].replace('\n', '').split()
            SitePos[0, 6] = float(content[0])
            SitePos[0, 7] = float(content[1])
            SitePos[0, 8] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            trip_clusters = np.vstack((trip_clusters, temparg_clust))

        elif temparg_clust[0, 2] == 4:

            SitePos = np.zeros([1, 12])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])
            content = file_info[kline + 1].replace('\n', '').split()
            SitePos[0, 3] = float(content[0])
            SitePos[0, 4] = float(content[1])
            SitePos[0, 5] = float(content[2])
            content = file_info[kline + 2].replace('\n', '').split()
            SitePos[0, 6] = float(content[0])
            SitePos[0, 7] = float(content[1])
            SitePos[0, 8] = float(content[2])
            content = file_info[kline + 3].replace('\n', '').split()
            SitePos[0, 9] = float(content[0])
            SitePos[0, 10] = float(content[1])
            SitePos[0, 11] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            quad_clusters = np.vstack((quad_clusters, temparg_clust))

        elif temparg_clust[0, 2] == 5:

            SitePos = np.zeros([1, 15])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])
            content = file_info[kline + 1].replace('\n', '').split()
            SitePos[0, 3] = float(content[0])
            SitePos[0, 4] = float(content[1])
            SitePos[0, 5] = float(content[2])
            content = file_info[kline + 2].replace('\n', '').split()
            SitePos[0, 6] = float(content[0])
            SitePos[0, 7] = float(content[1])
            SitePos[0, 8] = float(content[2])
            content = file_info[kline + 3].replace('\n', '').split()
            SitePos[0, 9] = float(content[0])
            SitePos[0, 10] = float(content[1])
            SitePos[0, 11] = float(content[2])
            content = file_info[kline + 4].replace('\n', '').split()
            SitePos[0, 12] = float(content[0])
            SitePos[0, 13] = float(content[1])
            SitePos[0, 14] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            site5_clusters = np.vstack((site5_clusters, temparg_clust))

        elif temparg_clust[0, 2] == 6:

            SitePos = np.zeros([1, 18])
            content = file_info[kline].replace('\n', '').split()
            SitePos[0, 0] = float(content[0])
            SitePos[0, 1] = float(content[1])
            SitePos[0, 2] = float(content[2])
            content = file_info[kline + 1].replace('\n', '').split()
            SitePos[0, 3] = float(content[0])
            SitePos[0, 4] = float(content[1])
            SitePos[0, 5] = float(content[2])
            content = file_info[kline + 2].replace('\n', '').split()
            SitePos[0, 6] = float(content[0])
            SitePos[0, 7] = float(content[1])
            SitePos[0, 8] = float(content[2])
            content = file_info[kline + 3].replace('\n', '').split()
            SitePos[0, 9] = float(content[0])
            SitePos[0, 10] = float(content[1])
            SitePos[0, 11] = float(content[2])
            content = file_info[kline + 4].replace('\n', '').split()
            SitePos[0, 12] = float(content[0])
            SitePos[0, 13] = float(content[1])
            SitePos[0, 14] = float(content[2])
            content = file_info[kline + 5].replace('\n', '').split()
            SitePos[0, 15] = float(content[0])
            SitePos[0, 16] = float(content[1])
            SitePos[0, 17] = float(content[2])

            temparg_clust = np.hstack((temparg_clust, SitePos))
            site6_clusters = np.vstack((site6_clusters, temparg_clust))

        # end of if
        kline += int(temparg_clust[0, 2])

        try:
            content = file_info[kline].replace('\n', '').split()
            if len(content) == 0: kline += 1
        except:
            print('  Finished reading clusters!')
        # end of try

    # end of while
    empty_cluster = empty_cluster[1:, :]
    point_cluster = point_cluster[1:, :]
    pair_clusters = pair_clusters[1:, :]
    trip_clusters = trip_clusters[1:, :]
    quad_clusters = quad_clusters[1:, :]
    site5_clusters = site5_clusters[1:, :]
    site6_clusters = site6_clusters[1:, :]

    NumClusters = np.zeros([7])
    NumClusters[0] = len(empty_cluster[:, 0])
    NumClusters[1] = len(point_cluster[:, 0])
    if len(pair_clusters[:, 0]) > 0: NumClusters[2] = len(pair_clusters[:, 0])
    if len(trip_clusters[:, 0]) > 0: NumClusters[3] = len(trip_clusters[:, 0])
    if len(quad_clusters[:, 0]) > 0: NumClusters[4] = len(quad_clusters[:, 0])
    if len(site5_clusters[:, 0]) > 0: NumClusters[5] = len(site5_clusters[:, 0])
    if len(site6_clusters[:, 0]) > 0: NumClusters[6] = len(site6_clusters[:, 0])

    NumClusters = NumClusters.astype('int32')

    ClusterInfo = np.vstack((empty_cluster[:, :3],
                             point_cluster[:, :3],
                             pair_clusters[:, :3],
                             trip_clusters[:, :3],
                             quad_clusters[:, :3],
                             site5_clusters[:, :3],
                             site6_clusters[:, :3]))

    MaxDiameter = np.zeros([7])
    if len(pair_clusters[:, 0]) > 0: MaxDiameter[2] = np.amax(pair_clusters[:, 1])
    if len(trip_clusters[:, 0]) > 0: MaxDiameter[3] = np.amax(trip_clusters[:, 1])
    if len(quad_clusters[:, 0]) > 0: MaxDiameter[4] = np.amax(quad_clusters[:, 1])
    if len(site5_clusters[:, 0]) > 0: MaxDiameter[5] = np.amax(site5_clusters[:, 1])
    if len(site6_clusters[:, 0]) > 0: MaxDiameter[6] = np.amax(site6_clusters[:, 1])

    ClusterCollection = {'NumClusters': NumClusters,
                         'ClusterInfo': ClusterInfo,
                         'MaxDiameter': MaxDiameter,
                         'empty_cluster': empty_cluster,
                         'point_cluster': point_cluster,
                         'pair_clusters': pair_clusters,
                         'trip_clusters': trip_clusters,
                         'quad_clusters': quad_clusters,
                         'site5_clusters': site5_clusters,
                         'site6_clusters': site6_clusters}

    return ClusterCollection

# -----------------------------------------------------------------------------

