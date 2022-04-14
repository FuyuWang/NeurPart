Constant KTileSz 2;
Constant CTileSz 1;
Constant ClusterSz 64;
Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 572, X: 572, R: 3, S: 3 }
Dataflow {
			SpatialMap(KTileSz,KTileSz) K;
			TemporalMap(ClusterSz,ClusterSz) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			Cluster(ClusterSz, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
}
}
}