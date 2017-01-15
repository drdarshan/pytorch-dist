#pragma once

#include "Storage.hpp"
#include "Type.hpp"

#include <TH/TH.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace thd {

// XXX: Forward declaration.
template<typename real> struct TensorScalarInterface;

struct Tensor {
  using long_range = std::vector<long>;

  Tensor() {};
  Tensor(const Tensor& other) = delete;
  Tensor(Tensor&& other) = delete;
  virtual ~Tensor() {};

  virtual Tensor* clone() const = 0;
  virtual Tensor* clone_shallow() = 0;

  virtual int nDim() const = 0;
  virtual long_range sizes() const = 0;
  virtual long_range strides() const = 0;
  virtual const long* rawSizes() const = 0;
  virtual const long* rawStrides() const = 0;
  virtual std::size_t storageOffset() const = 0;
  virtual std::size_t elementSize() const = 0;
  virtual long long numel() const = 0;
  virtual bool isContiguous() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual Tensor& retain() = 0;
  virtual Tensor& free() = 0;

  virtual Tensor& resize(const std::initializer_list<long>& new_size) = 0;
  virtual Tensor& resize(const std::vector<long>& new_size) = 0;
  virtual Tensor& resize(THLongStorage *size,
                         THLongStorage *stride) = 0;
  virtual Tensor& resizeAs(const Tensor& src) = 0;
  virtual Tensor& set(const Tensor& src) = 0;
  virtual Tensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             THLongStorage *size,
                             THLongStorage *stride) = 0;
  virtual Tensor& narrow(const Tensor& src,
                         int dimension,
                         long firstIndex,
                         long size) = 0;
  virtual Tensor& select(const Tensor& src, int dimension, long sliceIndex) = 0;
  virtual Tensor& transpose(const Tensor& src, int dimension1, int dimension2) = 0;
  virtual Tensor& unfold(const Tensor& src, int dimension, long size, long step) = 0;

  virtual Tensor& gather(const Tensor& src, int dimension, const Tensor& index) = 0;
  virtual Tensor& scatter(int dimension, const Tensor& index, const Tensor& src) = 0;
  virtual Tensor& neg(const Tensor& src) = 0;
  virtual Tensor& cinv(const Tensor& src) = 0;
  virtual Tensor& cmul(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cpow(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cdiv(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cfmod(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cremainder(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& max(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& min(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) = 0;
  virtual Tensor& mode(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& median(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& sum(const Tensor& src, int dimension) = 0;
  virtual Tensor& prod(const Tensor& src, int dimension) = 0;
  virtual Tensor& cumsum(const Tensor& src, int dimension) = 0;
  virtual Tensor& cumprod(const Tensor& src, int dimension) = 0;
  virtual Tensor& sign(const Tensor& source) = 0;
  virtual Tensor& cross(const Tensor& src1, const Tensor& src2, int dimension) = 0;
  virtual Tensor& cmax(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cmin(const Tensor& src1, const Tensor& src2) = 0;

  virtual std::unique_ptr<Tensor> newWithSize2d(long size0_, long size1_) = 0;
  virtual std::unique_ptr<Tensor> newWithSize3d(long size0_, long size1_, long size2_) = 0;
  virtual std::unique_ptr<Tensor> newWithSize4d(long size0_, long size1_, long size2_, long size3_) = 0;
  virtual std::unique_ptr<Tensor> newWithStorage(const Storage& storage_, ptrdiff_t storageOffset_, const StorageScalarInterface<long>& size_, const StorageScalarInterface<long>& stride_) = 0;
  virtual std::unique_ptr<Tensor> newWithStorage1d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_) = 0;
  virtual std::unique_ptr<Tensor> newWithStorage2d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_) = 0;
  virtual std::unique_ptr<Tensor> newWithStorage3d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_) = 0;
  virtual std::unique_ptr<Tensor> newWithStorage4d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_, long size3_, long stride3_) = 0;
  virtual std::unique_ptr<Tensor> newClone() = 0;
  virtual std::unique_ptr<Tensor> newContiguous() = 0;
  virtual std::unique_ptr<Tensor> newSelect(int dimension_, long sliceIndex_) = 0;
  virtual std::unique_ptr<Tensor> newNarrow(int dimension_, long firstIndex_, long size_) = 0;
  virtual std::unique_ptr<Tensor> newTranspose(int dimension1_, int dimension2_) = 0;
  virtual std::unique_ptr<Tensor> newUnfold(int dimension_, long size_, long step_) = 0;
  virtual Tensor& squeeze(const Tensor& src) = 0;
  virtual Tensor& squeeze1d(const Tensor& src, int dimension_) = 0;
  virtual int isContiguous() = 0;
  virtual int isSameSizeAs(const Tensor& src) = 0;
  virtual int isSetTo(const Tensor& src) = 0;
  virtual int isSize(const StorageScalarInterface<long>& dims) = 0;
  virtual ptrdiff_t nElement() = 0;
  virtual void gesv(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_) = 0;
  virtual void trtrs(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_, const char *uplo, const char *trans, const char *diag) = 0;
  virtual void gels(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_) = 0;
  virtual void syev(const Tensor& re_, const Tensor& rv_, const Tensor& a_, const char *jobz, const char *uplo) = 0;
  virtual void geev(const Tensor& re_, const Tensor& rv_, const Tensor& a_, const char *jobvr) = 0;
  virtual void gesvd(const Tensor& ru_, const Tensor& rs_, const Tensor& rv_, const Tensor& a, const char *jobu) = 0;
  virtual void gesvd2(const Tensor& ru_, const Tensor& rs_, const Tensor& rv_, const Tensor& ra_, const Tensor& a, const char *jobu) = 0;
  virtual void getri(const Tensor& ra_, const Tensor& a) = 0;
  virtual void potrf(const Tensor& ra_, const Tensor& a, const char *uplo) = 0;
  virtual void potrs(const Tensor& rb_, const Tensor& b_, const Tensor& a_,  const char *uplo) = 0;
  virtual void potri(const Tensor& ra_, const Tensor& a, const char *uplo) = 0;
  virtual void qr(const Tensor& rq_, const Tensor& rr_, Tensor *a) = 0;
  virtual void geqrf(const Tensor& ra_, const Tensor& rtau_, const Tensor& a) = 0;
  virtual void orgqr(const Tensor& ra_, const Tensor& a, const Tensor& tau) = 0;
  virtual void ormqr(const Tensor& ra_, const Tensor& a, const Tensor& tau, const Tensor& c, const char *side, const char *trans) = 0;
  virtual Tensor& zero() = 0;

  virtual thd::Type type() const = 0;
  virtual std::unique_ptr<Tensor> newTensor() const = 0;
};

template<typename real>
struct TensorScalarInterface : public Tensor {
  using scalar_type = real;
  virtual TensorScalarInterface& fill(scalar_type value) = 0;

  virtual TensorScalarInterface& scatterFill(int dimension, const Tensor& index, scalar_type value) = 0;
  virtual scalar_type dot(const Tensor& source) = 0;
  virtual scalar_type minall() = 0;
  virtual scalar_type maxall() = 0;
  virtual scalar_type sumall() = 0;
  virtual scalar_type prodall() = 0;
  virtual TensorScalarInterface& add(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& sub(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& mul(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& div(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& fmod(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& remainder(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) = 0;
  virtual TensorScalarInterface& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) = 0;
  virtual TensorScalarInterface& csub(const Tensor& src1, scalar_type value, const Tensor& src2) = 0;
  virtual TensorScalarInterface& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) = 0;
  virtual TensorScalarInterface& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) = 0;
  virtual TensorScalarInterface& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) = 0;
  virtual TensorScalarInterface& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) = 0;
  virtual TensorScalarInterface& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) = 0;
  virtual TensorScalarInterface& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) = 0;
  virtual TensorScalarInterface& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) = 0;
  virtual TensorScalarInterface& match(const Tensor& m1, const Tensor& m2, scalar_type gain) = 0;
  virtual scalar_type trace() = 0;
  virtual TensorScalarInterface& cmaxValue(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& cminValue(const Tensor& src, scalar_type value) = 0;

  virtual TensorScalarInterface<scalar_type> newWithSize1d(long size0_) = 0;

  virtual TensorScalarInterface& pstrf(const Tensor& ra_, const TensorScalarInterface<int>& rpiv_, const Tensor*& a, const char uplo, scalar_type tol) = 0;
  virtual TensorScalarInterface& fill(const Tensor& r_, scalar_type value) = 0;
  virtual TensorScalarInterface& maskedFill(const Tensor& tensor, const TensorScalarInterface<unsigned char>& mask, scalar_type value) = 0;

  virtual std::unique_ptr<Tensor> maskedCopy(const TensorScalarInterface<unsigned char>& mask, const Tensor& src) = 0;
  virtual std::unique_ptr<Tensor> maskedSelect(const Tensor& src, const TensorScalarInterface<unsigned char>& mask) = 0;
  virtual void nonzero(const TensorScalarInterface<long>& subscript, const Tensor& tensor) = 0; // TODO: Is this signature fine?
  virtual std::unique_ptr<Tensor> indexSelect(const Tensor& src, int dim, const TensorScalarInterface<long>& index) = 0;
  virtual std::unique_ptr<Tensor> indexCopy(int dim, const TensorScalarInterface<long>& index, const Tensor& src) = 0;
  virtual std::unique_ptr<Tensor> indexAdd(int dim, const TensorScalarInterface<long>& index, const Tensor& src) = 0;
  virtual std::unique_ptr<TensorScalarInterface> indexFill(int dim, const TensorScalarInterface<long>& index, scalar_type val) = 0;
};

using FloatTensor = TensorScalarInterface<double>;
using IntTensor = TensorScalarInterface<long long>;

} // namespace thd

