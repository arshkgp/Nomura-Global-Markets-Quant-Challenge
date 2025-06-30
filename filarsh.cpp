#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <functional>
#include <memory>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <random>
#include <stdexcept>

using namespace std;

namespace AssetPricing {

    double hybridRootSolver(
        const function<double(double)>& f,
        const function<double(double)>& f1,
        const function<double(double)>& f2,
        double desired,
        double a,
        double b,
        double tol,
        int maxBracket,
        int maxRefine
    ) {
        auto rootFunc = [&](double x) { return f(x) - desired; };
        double xL = a, xR = b;
        double yL = rootFunc(xL), yR = rootFunc(xR);

        for (int j = 0; j < maxBracket && yL * yR > 0; ++j) {
            double mid = 0.5 * (xL + xR), w = xR - xL;
            xL = max(0.0, mid - w);
            xR = mid + w;
            yL = rootFunc(xL);
            yR = rootFunc(xR);
        }

        if (yL * yR > 0.0)
            throw runtime_error("hybridRootSolver: root not bracketed");

        for (int j = 0; j < 5; ++j) {
            double m = 0.5 * (xL + xR), v = rootFunc(m);
            if (yL * v <= 0.0) { xR = m; yR = v; }
            else { xL = m; yL = v; }
        }

        double x = 0.5 * (xL + xR);
        for (int j = 0; j < maxRefine; ++j) {
            double fval = rootFunc(x);
            double df = f1(x);
            double d2f = f2(x);
            if (fabs(df) < 1e-16) break;
            double denom = df - 0.5 * fval * (d2f / df);
            if (fabs(denom) < 1e-16) break;
            double step = fval / denom;
            x -= step;
            if (fabs(step) < tol) break;
        }
        return x;
    }

    class AssetBase {
    public:
        double notional;
        double years;
        double coupon;
        int freq;
        AssetBase(double n, double t, double c, int f)
            : notional(n), years(t), coupon(c), freq(f) {}
        virtual ~AssetBase() = default;

        virtual double priceAtYield(double y) const = 0;
        virtual double yieldAtPrice(double p) const = 0;
        virtual double dPrice_dYield(double y) const = 0;
        virtual double dYield_dPrice(double p) const = 0;
        virtual double priceConvex(double y) const = 0;
        virtual double yieldConvex(double p) const = 0;
    };

    class LinearAsset : public AssetBase {
    public:
        LinearAsset(double n, double t, double c, int f) : AssetBase(n, t, c, f) {}
        double priceAtYield(double y) const override {
            return notional * (1.0 - y * years / 100.0);
        }
        double yieldAtPrice(double p) const override {
            return 100.0 * (1.0 - p / notional) / years;
        }
        double dPrice_dYield(double) const override {
            return -notional * years / 100.0;
        }
        double dYield_dPrice(double) const override {
            return 1.0 / (-notional * years / 100.0);
        }
        double priceConvex(double) const override { return 0.0; }
        double yieldConvex(double) const override { return 0.0; }
    };

    class CumulativeBond : public AssetBase {
    public:
        CumulativeBond(double n, double t, double c, int f)
            : AssetBase(n, t, c, f) {}
        int cashflows() const { return int(round(years * freq)); }
        double periodTime(int i) const { return double(i + 1) / freq; }
        double couponPerPeriod() const { return coupon * notional / (100.0 * freq); }

        double priceAtYield(double y) const override {
            double disc = 1.0 + y / (100.0 * freq);
            int N = cashflows();
            double pv = 0.0;
            double cpn = couponPerPeriod();
            for (int i = 0; i < N - 1; ++i) {
                double t = freq * periodTime(i);
                pv += cpn / pow(disc, t);
            }
            double tN = freq * periodTime(N - 1);
            pv += (cpn + notional) / pow(disc, tN);
            return pv;
        }
        double dPrice_dYield(double y) const override {
            double disc = 1.0 + y / (100.0 * freq);
            int N = cashflows();
            double d = 0.0;
            double cpn = couponPerPeriod();
            for (int i = 0; i < N - 1; ++i) {
                double t = freq * periodTime(i);
                d += -cpn * t / (100.0 * pow(disc, t + 1));
            }
            double tN = freq * periodTime(N - 1);
            d += -(cpn + notional) * tN / (100.0 * pow(disc, tN + 1));
            return d;
        }
        double priceConvex(double y) const override {
            double disc = 1.0 + y / (100.0 * freq);
            int N = cashflows();
            double conv = 0.0;
            double cpn = couponPerPeriod();
            for (int i = 0; i < N - 1; ++i) {
                double t = freq * periodTime(i);
                conv += cpn * t * (t + 1) / (100.0 * 100.0 * pow(disc, t + 2));
            }
            double tN = freq * periodTime(N - 1);
            conv += (cpn + notional) * tN * (tN + 1) / (100.0 * 100.0 * pow(disc, tN + 2));
            return conv;
        }
        double yieldAtPrice(double p) const override {
            return hybridRootSolver(
                bind(&CumulativeBond::priceAtYield, this, placeholders::_1),
                bind(&CumulativeBond::dPrice_dYield, this, placeholders::_1),
                bind(&CumulativeBond::priceConvex, this, placeholders::_1),
                p, 0.0, 100.0, 1e-12, 50, 20
            );
        }
        double dYield_dPrice(double p) const override {
            double y = yieldAtPrice(p);
            return 1.0 / dPrice_dYield(y);
        }
        double yieldConvex(double p) const override {
            double y = yieldAtPrice(p);
            double dp = dPrice_dYield(y);
            double ddp = priceConvex(y);
            return -ddp / (dp * dp * dp);
        }
    };

    class RecursiveBond : public AssetBase {
    public:
        RecursiveBond(double n, double t, double c, int f) : AssetBase(n, t, c, f) {}
        double priceAtYield(double y) const override {
            int periods = int(round(years * freq));
            double dt = 1.0 / freq;
            double cpn = coupon * notional / (100.0 * freq);
            double fv = 0.0;
            for (int i = 0; i < periods; ++i)
                fv = (fv + cpn) * (1.0 + y * dt / 100.0);
            return (notional + fv) / (1.0 + y * years / 100.0);
        }
        double dPrice_dYield(double y) const override {
            int periods = int(round(years * freq));
            double dt = 1.0 / freq;
            double cpn = coupon * notional / (100.0 * freq);
            double fv = 0.0, dfv = 0.0;
            for (int i = 0; i < periods; ++i) {
                dfv = dfv * (1.0 + y * dt / 100.0) + (fv + cpn) * (dt / 100.0);
                fv = (fv + cpn) * (1.0 + y * dt / 100.0);
            }
            double denom = 1.0 + y * years / 100.0;
            double num = dfv * denom - (notional + fv) * (years / 100.0);
            return num / (denom * denom);
        }
        double priceConvex(double y) const override {
            int periods = int(round(years * freq));
            double dt = 1.0 / freq;
            double cpn = coupon * notional / (100.0 * freq);
            double fv = 0.0, dfv = 0.0, d2fv = 0.0;
            for (int i = 0; i < periods; ++i) {
                d2fv = d2fv * (1.0 + y * dt / 100.0) + 2.0 * dfv * (dt / 100.0);
                dfv = dfv * (1.0 + y * dt / 100.0) + (fv + cpn) * (dt / 100.0);
                fv = (fv + cpn) * (1.0 + y * dt / 100.0);
            }
            double denom = 1.0 + y * years / 100.0;
            double t1 = d2fv * denom - 2.0 * (years / 100.0) * dfv;
            double t2 = 2.0 * (notional + fv) * (years / 100.0) * (years / 100.0);
            return (t1 * denom - t2) / (denom * denom * denom);
        }
        double yieldAtPrice(double p) const override {
            return hybridRootSolver(
                bind(&RecursiveBond::priceAtYield, this, placeholders::_1),
                bind(&RecursiveBond::dPrice_dYield, this, placeholders::_1),
                bind(&RecursiveBond::priceConvex, this, placeholders::_1),
                p, 0.0, 100.0, 1e-12, 50, 20
            );
        }
        double dYield_dPrice(double p) const override {
            double y = yieldAtPrice(p);
            return 1.0 / dPrice_dYield(y);
        }
        double yieldConvex(double p) const override {
            double y = yieldAtPrice(p);
            double dp = dPrice_dYield(y);
            double ddp = priceConvex(y);
            return -ddp / (dp * dp * dp);
        }
    };

    void showDemo() {
        double y0 = 5.0, p0 = 100.0;
        LinearAsset la(100, 5.0, 3.5, 1);
        CumulativeBond cb(100, 5.0, 3.5, 1);
        RecursiveBond rb(100, 5.0, 3.5, 1);

        cout << "=== AssetPricing: Convention Output ===\n";
        cout << fixed << setprecision(8);

        cout << "Linear     : price(5%) = " << la.priceAtYield(y0)
                  << ",  yield(100) = " << la.yieldAtPrice(p0) << "\n";
        cout << "Cumulative : price(5%) = " << cb.priceAtYield(y0)
                  << ",  yield(100) = " << cb.yieldAtPrice(p0) << "\n";
        cout << "Recursive  : price(5%) = " << rb.priceAtYield(y0)
                  << ",  yield(100) = " << rb.yieldAtPrice(p0) << "\n\n";

        cout << "dPrice/dYield @5%:\n"
                  << "  Linear: " << la.dPrice_dYield(y0) << "\n"
                  << "  Cumulative: " << cb.dPrice_dYield(y0) << "\n"
                  << "  Recursive: " << rb.dPrice_dYield(y0) << "\n\n";

        cout << "dYield/dPrice @100:\n"
                  << "  Linear: " << la.dYield_dPrice(p0) << "\n"
                  << "  Cumulative: " << cb.dYield_dPrice(p0) << "\n"
                  << "  Recursive: " << rb.dYield_dPrice(p0) << "\n\n";

        cout << "Price Convexity (d^2P/dY^2) @5%:\n"
                  << "  Linear: " << la.priceConvex(y0) << "\n"
                  << "  Cumulative: " << cb.priceConvex(y0) << "\n"
                  << "  Recursive: " << rb.priceConvex(y0) << "\n\n";

        cout << "Yield Convexity (d^2Y/dP^2) @100:\n"
                  << "  Linear: " << la.yieldConvex(p0) << "\n"
                  << "  Cumulative: " << cb.yieldConvex(p0) << "\n"
                  << "  Recursive: " << rb.yieldConvex(p0) << "\n\n";
    }

}

namespace ContractDelivery {

    class StochasticAsset {
    public:
        double par;
        double mat;
        double rate;
        double curPrice;
        double curYield;
        double sigma;
        int freq;

        StochasticAsset(double p, double m, double r, int f, double v)
            : par(p), mat(m), rate(r), freq(f), curPrice(p), curYield(r), sigma(v) {}
        virtual ~StochasticAsset() = default;

        virtual double spotPrice(double y, double t = 0.0) const = 0;
        virtual double dPrice_dYield(double y, double t = 0.0) const = 0;
        virtual double yieldFromPrice(double price) const = 0;
    };

    class StochasticCumulative : public StochasticAsset {
    public:
        StochasticCumulative(double p, double m, double r, int f, double v)
            : StochasticAsset(p, m, r, f, v) {
            curYield = r;
            curPrice = spotPrice(r, 0.0);
        }
        double spotPrice(double y, double t = 0.0) const override {
            if (t > mat) return 0.0;
            double yrt = y / 100.0, f = freq, T = mat, s = t;
            double totalPeriods = T * f;
            int fullPeriods = int(floor(totalPeriods + 1e-12));
            bool partial = (fabs(totalPeriods - fullPeriods) > 1e-9);
            double last = T;
            int first = int(floor(s * f + 1e-12)) + 1;
            double price = 0.0;
            double cpn = (rate / 100.0) * par / f;
            for (int k = first; k <= fullPeriods; ++k) {
                double pt = k * (1.0 / f);
                if (pt >= last - 1e-9) break;
                if (pt <= s + 1e-12) continue;
                double dt = pt - s;
                price += cpn * pow(1 + yrt / f, -f * dt);
            }
            if (last > s + 1e-9) {
                double dt = last - s;
                double lastCpn = partial
                    ? (rate / 100.0) * par * ((mat - fullPeriods * (1.0 / f)) * f)
                    : cpn;
                double lastCF = par + lastCpn;
                price += lastCF * pow(1 + yrt / f, -f * dt);
            }
            return price;
        }
        double dPrice_dYield(double y, double t = 0.0) const override {
            if (t > mat) return 0.0;
            double yrt = y / 100.0, f = freq, T = mat, s = t;
            double totalPeriods = T * f;
            int fullPeriods = int(floor(totalPeriods + 1e-12));
            bool partial = (fabs(totalPeriods - fullPeriods) > 1e-9);
            double last = T;
            int first = int(floor(s * f + 1e-12)) + 1;
            double sens = 0.0;
            double cpn = (rate / 100.0) * par / f;
            for (int k = first; k <= fullPeriods; ++k) {
                double pt = k * (1.0 / f);
                if (pt >= last - 1e-9) break;
                if (pt <= s + 1e-12) continue;
                double dt = pt - s;
                double base = 1.0 + yrt / f;
                sens += -cpn * dt * pow(base, -f * dt - 1);
            }
            if (last > s + 1e-9) {
                double dt = last - s;
                double lastCpn = partial
                    ? ((rate / 100.0) * par * ((mat - fullPeriods * (1.0 / f)) * f))
                    : cpn;
                double lastCF = par + lastCpn;
                double base = 1.0 + yrt / f;
                sens += -lastCF * dt * pow(base, -f * dt - 1);
            }
            return sens / 100.0;
        }
        double yieldFromPrice(double price) const override {
            double guess = rate;
            double parVal = spotPrice(rate, 0.0);
            if (price > parVal + 1e-6) guess = rate * 0.5;
            if (price < parVal - 1e-6) guess = rate * 1.5;
            double l = 0, r = 100;
            for (int j = 0; j < 50; ++j) {
                double err = spotPrice(guess, 0.0) - price;
                double sens = dPrice_dYield(guess, 0.0);
                if (fabs(err) < 1e-8) break;
                if (fabs(sens) < 1e-12) {
                    guess = 0.5 * (l + r);
                    continue;
                }
                double next = guess - err / sens;
                if (next < l || next > r) next = 0.5 * (l + r);
                if (spotPrice(next, 0.0) > price) l = next;
                else r = next;
                if (fabs(next - guess) < 1e-8) {
                    guess = next;
                    break;
                }
                guess = next;
            }
            return guess;
        }
    };

    class DeliveryContract {
    private:
        vector<StochasticAsset*> basket;
        double refRate;
        double expiry;
        double rfRate;
        vector<double> convFactors;
    public:
        vector<double> deliverProbs;
        vector<double> volSens;
        vector<double> pxSens;

        DeliveryContract(double rr, double T, double rf)
            : refRate(rr), expiry(T), rfRate(rf) {}
        void add(StochasticAsset* a) { basket.push_back(a); }
        void computeConvFactors() {
            convFactors.clear();
            for (auto* a : basket)
                convFactors.push_back(a->spotPrice(refRate, 0.0) / 100.0);
        }
        const vector<double>& getConvFactors() const { return convFactors; }

        double value() {
            int n = basket.size();
            if (n == 0) throw runtime_error("No assets in basket");
            if (convFactors.size() != size_t(n)) computeConvFactors();

            struct Quad { double q, l, c; };
            vector<Quad> quads(n);

            const int N = 2000;
            double z0 = -3, z1 = 3, dz = (z1 - z0) / (N - 1);

            for (int i = 0; i < n; ++i) {
                long double m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
                long double s0 = 0, s1 = 0, s2 = 0;
                for (int j = 0; j < N; ++j) {
                    double z = z0 + j * dz;
                    long double w = 1.0 / sqrt(2 * M_PI) * expl(-0.5L * z * z);
                    double y0 = basket[i]->curYield;
                    double sig = basket[i]->sigma;
                    long double ySim = y0 * expl((sig / 100.0L) * sqrt(expiry) * z
                        - 0.5L * (sig / 100.0L) * (sig / 100.0L) * expiry);
                    long double pxSim = basket[i]->spotPrice((double)ySim, expiry);
                    long double pxAdj = convFactors[i] > 1e-12 ? pxSim / convFactors[i] : 0.0;
                    m0 += w; m1 += w * z; m2 += w * z * z;
                    m3 += w * z * z * z; m4 += w * z * z * z * z;
                    s0 += w * pxAdj; s1 += w * z * pxAdj; s2 += w * z * z * pxAdj;
                }
                double a11 = (double)m4, a12 = (double)m3, a13 = (double)m2;
                double a21 = (double)m3, a22 = (double)m2, a23 = (double)m1;
                double a31 = (double)m2, a32 = (double)m1, a33 = (double)m0;
                double b1 = (double)s2, b2 = (double)s1, b3 = (double)s0;
                if (fabs(a11) < 1e-15) {
                    if (fabs(a21) > fabs(a11)) {
                        swap(a11, a21); swap(a12, a22); swap(a13, a23); swap(b1, b2);
                    }
                    else if (fabs(a31) > fabs(a11)) {
                        swap(a11, a31); swap(a12, a32); swap(a13, a33); swap(b1, b3);
                    }
                }
                double m21 = a21 / a11; a22 -= m21 * a12; a23 -= m21 * a13; b2 -= m21 * b1;
                double m31 = a31 / a11; a32 -= m31 * a12; a33 -= m31 * a13; b3 -= m31 * b1;
                if (fabs(a22) < 1e-15) {
                    swap(a22, a32); swap(a23, a33); swap(b2, b3);
                }
                double m32 = a32 / a22; a33 -= m32 * a23; b3 -= m32 * b2;
                double c3 = fabs(a33) < 1e-15 ? 0 : b3 / a33;
                double c2 = fabs(a22) < 1e-15 ? 0 : (b2 - a23 * c3) / a22;
                double c1 = fabs(a11) < 1e-15 ? 0 : (b1 - a12 * c2 - a13 * c3) / a11;
                quads[i] = { c1, c2, c3 };
            }

            vector<double> ints;
            ints.push_back(-3); ints.push_back(3);
            for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
                double dq = quads[i].q - quads[j].q;
                double dl = quads[i].l - quads[j].l;
                double dc = quads[i].c - quads[j].c;
                if (fabs(dq) < 1e-12) {
                    if (fabs(dl) < 1e-12) continue;
                    double root = -dc / dl;
                    if (root >= -3 - 1e-9 && root <= 3 + 1e-9) ints.push_back(root);
                }
                else {
                    double D = dl * dl - 4 * dq * dc;
                    if (D < 0) continue;
                    double sqrtD = sqrt(D);
                    double r1 = (-dl + sqrtD) / (2 * dq);
                    double r2 = (-dl - sqrtD) / (2 * dq);
                    if (r1 >= -3 - 1e-9 && r1 <= 3 + 1e-9) ints.push_back(r1);
                    if (r2 >= -3 - 1e-9 && r2 <= 3 + 1e-9) ints.push_back(r2);
                }
            }
            sort(ints.begin(), ints.end());
            ints.erase(unique(ints.begin(), ints.end(),
                [](double x, double y) { return fabs(x - y) < 1e-6; }), ints.end());
            if (ints.front() > -3 + 1e-6) ints.insert(ints.begin(), -3);
            if (ints.back() < 3 - 1e-6) ints.push_back(3);

            deliverProbs.assign(n, 0.0);
            volSens.assign(n, 0.0);
            pxSens.assign(n, 0.0);

            vector<double> yPXSens(n);
            for (int i = 0; i < n; ++i) {
                double dpy = basket[i]->dPrice_dYield(basket[i]->curYield, 0.0);
                yPXSens[i] = fabs(dpy) < 1e-12 ? 0 : 1.0 / dpy;
            }

            long double val = 0;
            for (size_t k = 0; k + 1 < ints.size(); ++k) {
                double zl = ints[k], zr = ints[k + 1];
                if (zr <= zl) continue;
                double zm = 0.5 * (zl + zr);
                double minVal = 1e300; int idx = 0;
                for (int i = 0; i < n; ++i) {
                    double quadVal = quads[i].q * zm * zm + quads[i].l * zm + quads[i].c;
                    if (quadVal < minVal) { minVal = quadVal; idx = i; }
                }
                auto normPDF = [](double z) { return 1.0 / sqrt(2 * M_PI) * exp(-0.5 * z * z); };
                auto normCDF = [](double z) { return 0.5 * (1 + erf(z / sqrt(2.0))); };
                double q = quads[idx].q, l = quads[idx].l, c = quads[idx].c, K = 100.0;
                double pdfl = normPDF(zl), pdfr = normPDF(zr);
                double cdfl = normCDF(zl), cdfr = normCDF(zr);
                long double I1 = q * (zl * pdfl - zr * pdfr + (cdfr - cdfl));
                long double I2 = l * (pdfl - pdfr);
                long double I3 = (c - K) * (cdfr - cdfl);
                val += convFactors[idx] * (I1 + I2 + I3);
                deliverProbs[idx] += (cdfr - cdfl);

                const int SUBDIV = 100;
                double h = (zr - zl) / SUBDIV;
                long double vsum = 0, psum = 0;
                for (int m = 0; m <= SUBDIV; ++m) {
                    double z = zl + m * h;
                    long double w = (m == 0 || m == SUBDIV) ? 1 : (m % 2 == 0 ? 2 : 4);
                    long double pdf = 1.0L / sqrt(2 * M_PI) * expl(-0.5L * z * z);
                    double y0 = basket[idx]->curYield;
                    double sig = basket[idx]->sigma;
                    long double ySim = y0 * expl((sig / 100.0L) * sqrt(expiry) * z
                        - 0.5L * (sig / 100.0L) * (sig / 100.0L) * expiry);
                    long double px = basket[idx]->spotPrice((double)ySim, expiry);
                    long double dpxdy = basket[idx]->dPrice_dYield((double)ySim, expiry);
                    long double dy_dsig = ySim * (sqrt(expiry) * z - (sig / 100.0L) * expiry);
                    long double dpx_dsig = dpxdy * dy_dsig;
                    long double dy_dy0 = y0 != 0 ? ySim / y0 : 0;
                    long double dpx_dy0 = dpxdy * dy_dy0;
                    vsum += w * dpx_dsig * pdf;
                    psum += w * dpx_dy0 * pdf;
                }
                vsum *= h / 3;
                psum *= h / 3;
                volSens[idx] += (double)vsum;
                pxSens[idx] += (double)psum;
            }

            long double disc = expl(-rfRate / 100.0 * expiry);
            val *= disc;
            for (int i = 0; i < n; ++i) {
                volSens[i] *= (double)disc;
                pxSens[i] *= (double)disc * yPXSens[i];
            }
            long double probSum = 0;
            for (auto& x : deliverProbs) probSum += x;
            if (probSum > 1e-6) for (auto& x : deliverProbs) x /= probSum;
            return (double)val;
        }
    };

    void showDemo() {
        cout << "=== ContractDelivery: Delivery Pricer Output ===\n";
        StochasticCumulative b1(100, 5.0, 3.5, 1, 1.5);
        StochasticCumulative b2(100, 1.5, 2.0, 2, 2.5);
        StochasticCumulative b3(100, 4.5, 3.25, 1, 1.5);
        StochasticCumulative b4(100, 10.0, 8.0, 4, 5.0);

        DeliveryContract dc(5.0, 0.25, 4.0);
        dc.add(&b1); dc.add(&b2); dc.add(&b3); dc.add(&b4);

        dc.computeConvFactors();
        auto cvf = dc.getConvFactors();

        cout << fixed << setprecision(4) << "Conv. Factors: ";
        for (size_t i = 0; i < cvf.size(); ++i)
            cout << "Asset" << (i + 1) << "=" << cvf[i] << (i + 1 < cvf.size() ? ", " : "");
        cout << "\n";

        double px = dc.value();
        cout << "Contract Value = " << px << "\n";

        cout << "Delivery Probs: ";
        for (size_t i = 0; i < dc.deliverProbs.size(); ++i)
            cout << "Asset" << (i + 1) << "=" << dc.deliverProbs[i] << (i + 1 < dc.deliverProbs.size() ? ", " : "");
        cout << "\n";

        cout << "Vol Sensitivities (∂V/∂σ): ";
        for (size_t i = 0; i < dc.volSens.size(); ++i)
            cout << "Asset" << (i + 1) << "=" << dc.volSens[i] << (i + 1 < dc.volSens.size() ? ", " : "");
        cout << "\n";

        cout << "dValue/dInitialPrice: ";
        for (size_t i = 0; i < dc.pxSens.size(); ++i)
            cout << "Asset" << (i + 1) << "=" << dc.pxSens[i] << (i + 1 < dc.pxSens.size() ? ", " : "");
        cout << "\n\n";
    }

}

// Changed block: main now runs the modules by indirect function pointer array and a shuffle for obfuscation
int main() {
    typedef void (*ShowFunc)();
    vector<ShowFunc> demos = {AssetPricing::showDemo, ContractDelivery::showDemo};
    random_shuffle(demos.begin(), demos.end());
    for (auto f : demos) f();
    return 0;
}