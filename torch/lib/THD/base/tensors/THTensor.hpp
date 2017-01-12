#pragma once

#include <TH/TH.h>

// We're defining THTensor as a custom class
#undef THTensor
#define THRealTensor TH_CONCAT_3(TH,Real,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thd {

template<typename real>
struct th_tensor_traits {};

#include "base/tensors/generic/THTensor.hpp"
#include <TH/THGenerateAllTypes.h>

} // namespace thd

#include "../storages/THStorage.hpp"

namespace thd {

template<typename real>
struct THTensor : public interface_traits<real>::tensor_interface_type {
  friend class THTensor<unsigned char>;
  friend class THTensor<char>;
  friend class THTensor<short>;
  friend class THTensor<int>;
  friend class THTensor<long>;
  friend class THTensor<float>;
  friend class THTensor<double>;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename th_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THTensor();
  THTensor(tensor_type *wrapped);
  virtual ~THTensor();

  virtual THTensor* clone() const override;
  virtual THTensor* clone_shallow() override;

  virtual int nDim() const override;
  virtual long_range sizes() const override;
  virtual long_range strides() const override;
  virtual const long* rawSizes() const override;
  virtual const long* rawStrides() const override;
  virtual std::size_t storageOffset() const override;
  virtual std::size_t elementSize() const override;
  virtual long long numel() const override;
  virtual bool isContiguous() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual THTensor& retain() override;
  virtual THTensor& free() override;

  virtual THTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THTensor& resize(const std::vector<long>& new_size) override;
  virtual THTensor& resize(THLongStorage *size,
                           THLongStorage *stride) override;
  virtual THTensor& resizeAs(const Tensor& src) override;
  virtual THTensor& set(const Tensor& src) override;
  virtual THTensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             THLongStorage *size,
                             THLongStorage *stride) override;

  virtual THTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THTensor& select(const Tensor& src, int dimension,
                           long sliceIndex) override;
  virtual THTensor& transpose(const Tensor& src, int dimension1,
                              int dimension2) override;
  virtual THTensor& unfold(const Tensor& src, int dimension,
                           long size, long step) override;

  virtual THTensor& fill(scalar_type value) override;

  virtual THTensor& gather(const Tensor& src, int dimension, const Tensor& index) override;
  virtual THTensor& scatter(int dimension, const Tensor& index, const Tensor& src) override;
  virtual THTensor& scatterFill(int dimension, const Tensor& index, scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THTensor& neg(const Tensor& src) override;
  virtual THTensor& cinv(const Tensor& src) override;
  virtual THTensor& add(const Tensor& src, scalar_type value) override;
  virtual THTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THTensor& div(const Tensor& src, scalar_type value) override;
  virtual THTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THTensor& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) override;
  virtual THTensor& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THTensor& csub(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) override;
  virtual THTensor& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) override;
  virtual THTensor& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) override;
  virtual THTensor& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THTensor& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THTensor& match(const Tensor& m1, const Tensor& m2, scalar_type gain) override;
  virtual THTensor& max(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& min(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) override;
  virtual THTensor& mode(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& median(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& sum(const Tensor& src, int dimension) override;
  virtual THTensor& prod(const Tensor& src, int dimension) override;
  virtual THTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THTensor& cross(const Tensor& src1, const Tensor& src2, int dimension) override;
  virtual THTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THTensor& cminValue(const Tensor& src, scalar_type value) override;

  virtual thd::Type type() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

  virtual THTensor *newWithSize1d(long size0_) override;
  virtual THTensor *newWithSize2d(long size0_, long size1_) override;
  virtual THTensor *newWithSize3d(long size0_, long size1_, long size2_) override;
  virtual THTensor *newWithSize4d(long size0_, long size1_, long size2_, long size3_) override;
  virtual THTensor *newWithStorage(THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_) override;
  virtual THTensor *newWithStorage1d(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_) override;
  virtual THTensor *newWithStorage2d(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_) override;
  virtual THTensor *newWithStorage3d(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_) override;
  virtual THTensor *newWithStorage4d(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_, long size3_, long stride3_) override;
  virtual THTensor *newClone(THTensor *self) override;
  virtual THTensor *newContiguous(THTensor *tensor) override;
  virtual THTensor *newSelect(THTensor *tensor, int dimension_, long sliceIndex_) override;
  virtual THTensor *newNarrow(THTensor *tensor, int dimension_, long firstIndex_, long size_) override;
  virtual THTensor *newTranspose(THTensor *tensor, int dimension1_, int dimension2_) override;
  virtual THTensor *newUnfold(THTensor *tensor, int dimension_, long size_, long step_) override;
  virtual void narrow(THTensor *self, THTensor *src, int dimension_, long firstIndex_, long size_) override;
  virtual void select(THTensor *self, THTensor *src, int dimension_, long sliceIndex_) override;
  virtual void transpose(THTensor *self, THTensor *src, int dimension1_, int dimension2_) override;
  virtual void unfold(THTensor *self, THTensor *src, int dimension_, long size_, long step_) override;
  virtual void squeeze(THTensor *self, THTensor *src) override;
  virtual void squeeze1d(THTensor *self, THTensor *src, int dimension_) override;
  virtual int isContiguous(const THTensor *self) override;
  virtual int isSameSizeAs(const THTensor *self, const THTensor *src) override;
  virtual int isSetTo(const THTensor *self, const THTensor *src) override;
  virtual int isSize(const THTensor *self, const THLongStorage *dims) override;
  virtual ptrdiff_t nElement(const THTensor *self) override;
  virtual void gesv(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_) override;
  virtual void trtrs(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag) override;
  virtual void gels(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_) override;
  virtual void syev(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo) override;
  virtual void geev(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr) override;
  virtual void gesvd(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu) override;
  virtual void gesvd2(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu) override;
  virtual void getri(THTensor *ra_, THTensor *a) override;
  virtual void potrf(THTensor *ra_, THTensor *a, const char *uplo) override;
  virtual void potrs(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo) override;
  virtual void potri(THTensor *ra_, THTensor *a, const char *uplo) override;
  virtual void qr(THTensor *rq_, THTensor *rr_, THTensor *a) override;
  virtual void geqrf(THTensor *ra_, THTensor *rtau_, THTensor *a) override;
  virtual void orgqr(THTensor *ra_, THTensor *a, THTensor *tau) override;
  virtual void ormqr(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans) override;
  virtual void pstrf(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol) override;
  virtual void fill(THTensor *r_, real value) override;
  virtual void zero(THTensor *r_) override;
  virtual void maskedFill(THTensor *tensor, THByteTensor *mask, real value) override;
  virtual void maskedCopy(THTensor *tensor, THByteTensor *mask, THTensor* src) override;
  virtual void maskedSelect(THTensor *tensor, THTensor* src, THByteTensor *mask) override;
  virtual void nonzero(THLongTensor *subscript, THTensor *tensor) override;
  virtual void indexSelect(THTensor *tensor, THTensor *src, int dim, THLongTensor *index) override;
  virtual void indexCopy(THTensor *tensor, int dim, THLongTensor *index, THTensor *src) override;
  virtual void indexAdd(THTensor *tensor, int dim, THLongTensor *index, THTensor *src) override;
  virtual void indexFill(THTensor *tensor, int dim, THLongTensor *index, real val) override;

private:
  template<typename iterator>
  THTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THTensor& resize(const iterator& size_begin, const iterator& size_end,
                   const iterator& stride_begin, const iterator& stride_end);

protected:
  tensor_type *tensor;
};

}
