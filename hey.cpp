#include <iostream>
#include <vector>
using namespace std;

class DisjointSet {
private:
    vector<int> parent, rank;

public:
    DisjointSet(int n) {
        parent.resize(n + 1);
        rank.resize(n + 1, 0);
        for (int i = 1; i <= n; i++)
            parent[i] = i;
    }

    int findParent(int x) {
        if (parent[x] != x)
            parent[x] = findParent(parent[x]);
        return parent[x];
    }

    void unionSets(int a, int b) {
        a = findParent(a);
        b = findParent(b);
        if (a != b) {
            if (rank[a] < rank[b])
                parent[a] = b;
            else if (rank[b] < rank[a])
                parent[b] = a;
            else {
                parent[b] = a;
                rank[a]++;
            }
        }
    }

    void printParents() {
        cout << "Parent array: ";
        for (int i = 1; i < parent.size(); i++)
            cout << parent[i] << " ";
        cout << "\n";
    }
};

int main() {
    cout << "===== DISJOINT SET IMPLEMENTATION =====\n";

    DisjointSet ds(5);

    cout << "Union(1, 2)\n";
    ds.unionSets(1, 2);
    ds.printParents();

    cout << "Union(3, 4)\n";
    ds.unionSets(3, 4);
    ds.printParents();

    cout << "Union(2, 4)\n";
    ds.unionSets(2, 4);
    ds.printParents();

    cout << "Find(3) gives parent: " << ds.findParent(3) << "\n";
    cout << "Find(4) gives parent: " << ds.findParent(4) << "\n";

    return 0;
}
